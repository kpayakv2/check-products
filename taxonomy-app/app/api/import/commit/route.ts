import { NextRequest, NextResponse } from 'next/server'
import { z } from 'zod'
import { supabaseAdmin } from '@/utils/supabase-admin'
import { rateLimit } from '@/utils/rate-limit'
import { withErrorHandling } from '@/utils/error-handler'
import { validateRequest } from '@/utils/validation'

/**
 * บันทึกผลการนำเข้าสินค้าใหม่ลงฐานข้อมูล
 *
 * แยกเป็นสองจังหวะเพราะ wizard ตัดสินใจคนละขั้น:
 *   action='dedup'      จบขั้นตรวจของซ้ำ → สร้าง batch + products ตามกลุ่ม
 *   action='categorize' จบขั้นจัดหมวด    → เติม category_id ให้ของใหม่
 *
 * ต้องบันทึกตั้งแต่จบขั้นตรวจของซ้ำ ไม่ใช่รอขั้นสุดท้าย ไม่งั้นถ้าปิดเบราว์เซอร์กลางคัน
 * รายการก้ำกึ่งที่ยังตรวจไม่จบจะหายทั้งหมด — รายการเหล่านี้ต้องไปโผล่ที่หน้า
 * Data Quality → Verify ได้ด้วย เพื่อให้ทำต่อจากที่ค้างไว้
 *
 * สถานะที่ใช้ (ตรงกับที่ VerifyTab อ่าน):
 *   rejected                มีในสตอกอยู่แล้ว ไม่ต้องเพิ่ม — เก็บไว้เป็นหลักฐานและข้อมูลฝึก ML
 *   pending_review_dedup    ก้ำกึ่ง รอคนตัดสินว่าซ้ำหรือไม่
 *   pending_review_category ของใหม่แน่ รอยืนยันหมวดหมู่
 */

const BUCKET_STATUS = {
  duplicate: 'rejected',
  review: 'pending_review_dedup',
  new: 'pending_review_category',
} as const

const ItemSchema = z.object({
  name_th: z.string().min(1, 'ต้องมีชื่อสินค้า'),
  cleaned_name: z.string().optional(),
  bucket: z.enum(['duplicate', 'review', 'new']),
  similarity: z.coerce.number().min(0).max(1).default(0),
  // สินค้าในสตอกที่จับคู่ได้ — ใช้สร้างคู่ใน similarity_matches
  matched_product_id: z.string().uuid().optional(),
  sku: z.string().optional(),
  price: z.coerce.number().optional(),
})

// ไอดีประจำหนึ่งรอบ wizard — ไม่บังคับเป็น uuid เพราะฝั่งเบราว์เซอร์มีทางสำรอง
// สำหรับเครื่อง LAN ที่ crypto.randomUUID ใช้ไม่ได้ (ไม่ใช่ secure context)
const WizardRunId = z.string().min(8).max(128)

const DedupCommitSchema = z.object({
  action: z.literal('dedup'),
  wizard_run_id: WizardRunId,
  file_name: z.string().optional(),
  items: z.array(ItemSchema).min(1, 'ต้องมีอย่างน้อย 1 รายการ').max(5000),
})

const CategorizeCommitSchema = z.object({
  action: z.literal('categorize'),
  wizard_run_id: WizardRunId.optional(),
  import_batch_id: z.string().uuid(),
  assignments: z.array(z.object({
    product_id: z.string().uuid(),
    category_id: z.string().uuid(),
    // ชื่อหมวด — VerifyTab อ่านจาก metadata.suggested_category เพื่อแสดงว่า "AI แนะนำ: ..."
    category_name: z.string().optional(),
    confidence_score: z.coerce.number().min(0).max(1).default(0),
  })).min(1).max(5000),
})

const CommitSchema = z.discriminatedUnion('action', [DedupCommitSchema, CategorizeCommitSchema])

const limiter = rateLimit({ interval: 60 * 1000, uniqueTokenPerInterval: 500 })

const INSERT_BATCH = 200
const EMBED_BATCH = 200

/**
 * ขอ embedding เป็นชุดจาก FastAPI
 *
 * ห้ามข้ามขั้นนี้: สินค้าที่ไม่มี embedding จะมองไม่เห็นในการสแกนหาของซ้ำครั้งต่อไป
 * แล้วจะถูกนำเข้าซ้ำได้อีกเรื่อยๆ เป็นปัญหาที่สะสมทบขึ้นทุกรอบ
 */
async function embedAll(texts: string[]): Promise<(number[] | null)[]> {
  const fastapiUrl = process.env.FASTAPI_URL || 'http://127.0.0.1:8000'
  const vectors: (number[] | null)[] = []

  for (let start = 0; start < texts.length; start += EMBED_BATCH) {
    const chunk = texts.slice(start, start + EMBED_BATCH)
    try {
      const response = await fetch(`${fastapiUrl}/api/embed/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts: chunk }),
      })
      if (!response.ok) throw new Error(`embed failed: ${response.status}`)
      const payload = await response.json()
      vectors.push(...(payload.embeddings as number[][]))
    } catch (error) {
      console.error('Embedding batch failed — จะบันทึกโดยไม่มี embedding:', error)
      vectors.push(...chunk.map(() => null))
    }
  }
  return vectors
}

/**
 * ประกอบผลลัพธ์ของ batch ที่บันทึกไปแล้วขึ้นมาใหม่
 *
 * ใช้ตอบเมื่อพบว่ารอบนี้เคยบันทึกไปแล้ว ต้องคืนคีย์ `products` ให้ครบเหมือนการบันทึกครั้งแรก
 * เพราะขั้นจัดหมวดหมู่เอาไปทำ map ชื่อ → id ถ้าขาดไปหมวดหมู่จะไม่ถูกบันทึกทั้งรอบ
 */
async function existingBatchResponse(batchId: string) {
  const { data: products, error } = await supabaseAdmin
    .from('products')
    .select('id, name_th, status')
    .eq('import_batch_id', batchId)
  if (error) throw error

  const rows = products ?? []
  const counts = rows.reduce<Record<string, number>>((acc, row) => {
    acc[row.status] = (acc[row.status] ?? 0) + 1
    return acc
  }, {})

  return NextResponse.json({
    success: true,
    import_batch_id: batchId,
    saved: rows.length,
    counts,
    // รอบนี้บันทึกไปแล้ว ไม่ได้เขียนอะไรเพิ่ม — บอกไว้เพื่อให้ debug ย้อนหลังได้
    reused: true,
    products: rows.map((row) => ({ id: row.id, name_th: row.name_th, status: row.status })),
  })
}

async function commitDedup(body: z.infer<typeof DedupCommitSchema>) {
  const { items, file_name, wizard_run_id } = body

  // กันบันทึกซ้ำก่อนทำงานหนัก — ผู้ใช้กดย้อน step กลับมาแล้วเดินหน้าใหม่ได้
  // ถ้าปล่อยผ่านจะได้ imports ใหม่ทั้งใบและ products ซ้ำทุกแถว
  // ซึ่งย้อนกลับไปทำให้การตรวจของซ้ำรอบหน้าเพี้ยนตามไปด้วย
  const { data: priorBatch } = await supabaseAdmin
    .from('imports')
    .select('id')
    .eq('metadata->>wizard_run_id', wizard_run_id)
    .maybeSingle()
  if (priorBatch) return existingBatchResponse(priorBatch.id)

  const embeddings = await embedAll(items.map((item) => item.name_th))
  const embeddedCount = embeddings.filter(Boolean).length

  // ดึงชื่อสินค้าในสตอกที่จับคู่ได้ เพื่อให้หน้า Verify แสดงว่า "คล้ายกับตัวไหน"
  const matchedIds = [...new Set(items.map((item) => item.matched_product_id).filter(Boolean))] as string[]
  const matchedNames = new Map<string, string>()
  if (matchedIds.length > 0) {
    const { data } = await supabaseAdmin
      .from('products')
      .select('id, name_th')
      .in('id', matchedIds)
    for (const row of data ?? []) matchedNames.set(row.id, row.name_th)
  }

  const { data: batch, error: batchError } = await supabaseAdmin
    .from('imports')
    .insert({
      name: `New products import - ${new Date().toISOString().slice(0, 16).replace('T', ' ')}`,
      description: 'นำเข้าสินค้าใหม่ผ่าน Import Wizard (ตรวจกับสตอกแล้ว)',
      file_name: file_name ?? null,
      total_records: items.length,
      status: 'processing',
      // ต้องเขียนตั้งแต่ตอน insert ไม่ใช่ตอน update ท้ายฟังก์ชัน
      // เพราะช่วง embedAll ข้างบนกินเวลานานพอให้คำขอที่สองแทรกเข้ามาได้
      metadata: { wizard_run_id },
    })
    .select()
    .single()

  if (batchError) {
    // แข่งกันเข้ามาพร้อมกันจน unique index ตัดออก — อีกคำขอบันทึกสำเร็จไปแล้ว
    // คืนผลของ batch นั้นแทน ไม่ใช่โยน error ให้ผู้ใช้เห็นว่าบันทึกไม่สำเร็จ
    if (batchError.code === '23505') {
      const { data: raced } = await supabaseAdmin
        .from('imports')
        .select('id')
        .eq('metadata->>wizard_run_id', wizard_run_id)
        .maybeSingle()
      if (raced) return existingBatchResponse(raced.id)
    }
    throw batchError
  }

  const rows = items.map((item, index) => ({
    name_th: item.name_th,
    sku: item.sku || null,
    price: item.price ?? null,
    embedding: embeddings[index],
    status: BUCKET_STATUS[item.bucket],
    confidence_score: item.similarity,
    import_batch_id: batch.id,
    // ชื่อ key ต้องตรงกับที่ VerifyTab อ่าน (clean_name / similarity_score / duplicate_of)
    // ไม่งั้นหน้าจะขึ้น "Match Score: NaN%" และช่องชื่อว่าง
    metadata: {
      source: 'import_wizard',
      clean_name: item.cleaned_name ?? item.name_th,
      similarity_score: item.similarity,
      duplicate_of: item.matched_product_id ? matchedNames.get(item.matched_product_id) ?? null : null,
      dedup_bucket: item.bucket,
      matched_product_id: item.matched_product_id ?? null,
    },
  }))

  const inserted: { id: string; name_th: string; status: string }[] = []
  for (let start = 0; start < rows.length; start += INSERT_BATCH) {
    const { data, error } = await supabaseAdmin
      .from('products')
      .insert(rows.slice(start, start + INSERT_BATCH))
      .select('id, name_th, status')
    if (error) throw error
    inserted.push(...(data ?? []))
  }

  // คู่ที่ระบบมั่นใจว่าซ้ำ — เก็บไว้เป็นหลักฐานและเป็นข้อมูลฝึก ML
  // ต้องสร้าง product ฝั่งใหม่ด้วย (สถานะ rejected) เพราะ similarity_matches
  // มี FK บังคับทั้งสองฝั่งว่าต้องมีแถวจริงใน products
  const byName = new Map(inserted.map((row) => [row.name_th, row.id]))
  const pairs = items
    .filter((item) => item.bucket === 'duplicate' && item.matched_product_id && byName.has(item.name_th))
    .map((item) => ({
      product_a_id: byName.get(item.name_th)!,
      product_b_id: item.matched_product_id!,
      similarity_score: item.similarity,
      match_type: 'import_duplicate',
      algorithm: 'import_wizard_dedup',
      is_duplicate: true,
      reviewed: true,
      reviewed_at: new Date().toISOString(),
      metadata: { product_a_name: item.name_th, labelled_by: 'threshold_import' },
    }))

  if (pairs.length > 0) {
    const { error } = await supabaseAdmin.from('similarity_matches').insert(pairs)
    if (error) throw error
  }

  const counts = inserted.reduce<Record<string, number>>((acc, row) => {
    acc[row.status] = (acc[row.status] ?? 0) + 1
    return acc
  }, {})

  await supabaseAdmin
    .from('imports')
    .update({
      status: 'completed',
      processed_records: inserted.length,
      success_records: inserted.length,
      error_records: items.length - inserted.length,
      // ต้องคง wizard_run_id ไว้ — update ทับ metadata ทั้งก้อน ถ้าลืมใส่คีย์จะหาย
      // แล้วการกันบันทึกซ้ำจะใช้ไม่ได้ตั้งแต่รอบแรก
      metadata: { wizard_run_id, counts, similarity_pairs: pairs.length, embedded: embeddedCount },
    })
    .eq('id', batch.id)

  return NextResponse.json({
    success: true,
    import_batch_id: batch.id,
    saved: inserted.length,
    counts,
    similarity_pairs: pairs.length,
    // ถ้า FastAPI ล่ม จะบันทึกได้แต่ไม่มี embedding — ต้องบอกให้รู้ ไม่ใช่เงียบ
    embedded: embeddedCount,
    missing_embedding: inserted.length - embeddedCount,
    products: inserted.map((row) => ({ id: row.id, name_th: row.name_th, status: row.status })),
  })
}

async function commitCategorize(body: z.infer<typeof CategorizeCommitSchema>) {
  const { import_batch_id, assignments } = body

  let updated = 0
  for (const assignment of assignments) {
    // อ่าน metadata เดิมมาก่อนแล้วเติม suggested_category ไม่ให้ทับของที่ commitDedup ใส่ไว้
    const { data: existing } = await supabaseAdmin
      .from('products')
      .select('metadata')
      .eq('id', assignment.product_id)
      .eq('import_batch_id', import_batch_id)
      .single()

    if (!existing) continue

    const { error } = await supabaseAdmin
      .from('products')
      .update({
        category_id: assignment.category_id,
        confidence_score: assignment.confidence_score,
        metadata: {
          ...(existing.metadata as Record<string, unknown>),
          suggested_category: assignment.category_name ?? null,
        },
        updated_at: new Date().toISOString(),
      })
      .eq('id', assignment.product_id)
      .eq('import_batch_id', import_batch_id)
    if (!error) updated += 1
  }

  const suggestions = assignments.map((assignment) => ({
    product_id: assignment.product_id,
    suggested_category_id: assignment.category_id,
    confidence_score: assignment.confidence_score,
    suggestion_method: 'import_wizard',
    metadata: { import_batch_id },
  }))

  // การ update products ข้างบนเขียนทับค่าเดิมอยู่แล้วจึงเรียกซ้ำได้
  // แต่ตารางนี้เป็น insert ล้วน ต้องล้างของรอบเดียวกันทิ้งก่อน ไม่งั้นกดซ้ำแล้วได้ข้อเสนอซ้ำ
  await supabaseAdmin
    .from('product_category_suggestions')
    .delete()
    .eq('suggestion_method', 'import_wizard')
    .eq('metadata->>import_batch_id', import_batch_id)

  for (let start = 0; start < suggestions.length; start += INSERT_BATCH) {
    await supabaseAdmin
      .from('product_category_suggestions')
      .insert(suggestions.slice(start, start + INSERT_BATCH))
  }

  return NextResponse.json({ success: true, updated, suggestions: suggestions.length })
}

export async function POST(request: NextRequest): Promise<NextResponse> {
  return withErrorHandling(async () => {
    try {
      await limiter.check(30, 'import-commit')
    } catch {
      return NextResponse.json(
        { success: false, error: 'ส่งคำขอถี่เกินไป กรุณารอสักครู่' },
        { status: 429 }
      )
    }

    const body = await validateRequest(request, CommitSchema)
    if ('error' in body) {
      return NextResponse.json(body, { status: 400 })
    }

    return body.action === 'dedup'
      ? commitDedup(body)
      : commitCategorize(body)
  })
}
