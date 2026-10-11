import { NextRequest, NextResponse } from 'next/server'
import { z } from 'zod'
import { supabaseAdmin } from '@/utils/supabase-admin'
import { withErrorHandling } from '@/utils/error-handler'
import {
  buildCategoryChain,
  buildPosExportFile,
  countPosExportIssues,
  toPosExportRow,
  type ExportableProduct,
  type PosExportRow,
  type TaxonomyNodeLite,
} from '@/utils/pos-export'
import { allocateItemIds } from '@/utils/pos-item-id'

const PREVIEW_LIMIT = 50

/** จำนวนแถวที่ส่งกลับให้ UI ดูก่อนตัดสินใจโหลด — ไม่ใช่จำนวนที่จะอยู่ในไฟล์จริง */
export const dynamic = 'force-dynamic'

interface ExportProductRow extends ExportableProduct {
  id: string
}

/**
 * ขนาดหน้าที่ไล่ดึงทีละรอบ — เท่ากับ `max_rows` ใน supabase/config.toml
 *
 * PostgREST ตัดผลลัพธ์ที่ไม่ใส่ range ทิ้งที่ max_rows แล้วตอบ 206 เฉย ๆ
 * supabase-js ไม่ถือว่านั่นเป็น error ด้วย (error เป็น null) โค้ดจึงเดินต่อ
 * ด้วยข้อมูลที่ขาดหายโดยไม่มีสัญญาณอะไรเลย
 */
const FETCH_PAGE_SIZE = 1000

/** query ที่ยังรับ .range() ได้ — พอสำหรับ helper นี้ ไม่ต้องลาก type ของ supabase-js มาทั้งก้อน */
interface RangeableQuery {
  range(from: number, to: number): PromiseLike<{ data: unknown[] | null; error: unknown }>
}

/**
 * ดึงทุกแถวที่ตรงเงื่อนไข โดยไล่ทีละหน้าจนกว่าจะได้หน้าที่ไม่เต็ม
 *
 * รับเป็นฟังก์ชันสร้าง query ไม่ใช่ตัว query เพราะ builder ของ supabase-js
 * ใช้ซ้ำหลายรอบไม่ได้ — ต้องประกอบใหม่ทุกหน้า
 *
 * ผู้เรียกต้องใส่ .order() ที่เสถียรมาด้วย ไม่งั้นลำดับของแต่ละหน้าไม่แน่นอน
 * แล้วบางแถวจะถูกข้ามหรือซ้ำ (ถ้ามีคนแทรกแถวคั่นกลางระหว่างหน้า ยังคลาดได้อยู่
 * แต่นั่นเป็นการแข่งกันเขียนจริง ๆ ไม่ใช่ความไม่แน่นอนของตัวคำสั่งเอง)
 */
async function fetchAllRows<T>(buildQuery: () => RangeableQuery): Promise<T[]> {
  const all: T[] = []

  for (let page = 0; ; page += 1) {
    const from = page * FETCH_PAGE_SIZE
    const { data, error } = await buildQuery().range(from, from + FETCH_PAGE_SIZE - 1)
    if (error) throw error

    const batch = (data ?? []) as T[]
    all.push(...batch)
    if (batch.length < FETCH_PAGE_SIZE) break
  }

  return all
}

/**
 * ดึงสินค้าที่ยังไม่ได้ส่งเข้า POS พร้อมแปลงเป็นแถวของไฟล์
 *
 * คัดด้วย metadata ไม่ใช่ status เพราะ /api/verify เซ็ต status เป็น 'approved'
 * ตอนคนกดยืนยันหมวด ซึ่งเป็นค่าเดียวกับสตอกเดิม 3,103 รายการ
 * metadata.source เป็นตัวเดียวที่ยังแยก 'import_wizard' ออกจาก 'legacy_labelled' ได้
 */
async function loadExportRows(): Promise<{
  ids: string[]
  rows: PosExportRow[]
  assignedSkus: Map<string, string>
}> {
  // ต้องไล่ทีละหน้า ไม่ใช่ยิงรวดเดียว — คิวเกิน 1,000 เมื่อไหร่ ไฟล์จะขาดแถว
  // และ total ที่โชว์บนหน้าจอจะน้อยกว่าของจริงโดยไม่มี error ให้จับ
  const list = await fetchAllRows<ExportProductRow>(() =>
    supabaseAdmin
      .from('products')
      .select('id, name_th, sku, barcode, unit_name, vat_flag, price, category_id')
      .eq('metadata->>source', 'import_wizard')
      .eq('metadata->>dedup_bucket', 'new')
      .is('pos_exported_at', null)
      .order('name_th')
      .order('id')
  )

  // taxonomy มีแค่ 134 โหนด แต่ไล่ทีละหน้าเหมือนกัน เผื่อวันหนึ่งโตเกิน max_rows
  // ดึงมาทั้งก้อนแล้วไล่สายในหน่วยความจำ ถูกกว่ายิงหา parent ทีละรายการ
  // และเลี่ยง raw SQL ตามกฎรีโป
  const nodes = await fetchAllRows<TaxonomyNodeLite>(() =>
    supabaseAdmin.from('taxonomy_nodes').select('id, name_th, parent_id').order('id')
  )

  // ต้องรู้ sku ทั้งหมดที่ใช้ไปแล้ว ไม่ใช่แค่เลขสูงสุด — products.sku มี UNIQUE
  // constraint ถ้าเลขในฐานข้อมูลขาดช่วง การเดินจาก max อย่างเดียวยังชนได้
  //
  // สตอกเดิมมี 3,103 แถวอยู่แล้ว การยิงรวดเดียวจึงได้มาแค่ 1,000 แถวแรกเสมอ
  // prefix ที่เลขสูงสุดอยู่นอกหน้าต่างนั้นจะถูกคำนวณต่ำเกินจริง แล้วออกรหัสชนของเดิม
  const usedSkus = await fetchAllRows<{ sku: string | null }>(() =>
    supabaseAdmin.from('products').select('sku').not('sku', 'is', null).order('sku')
  )

  const nodesById = new Map<string, TaxonomyNodeLite>(
    nodes.map((node) => [node.id, node])
  )

  // แจกรหัสทั้งใน GET และ POST ด้วยข้อมูลชุดเดียวกัน ต่างกันแค่ POST เขียนลงฐานข้อมูล
  // ตัวแจกเรียงตามชื่อเสมอ รหัสที่เห็นในหน้าตัวอย่างจึงตรงกับไฟล์ที่โหลดจริง
  const assignedSkus = allocateItemIds(
    list.map((product) => ({
      id: product.id,
      name_th: product.name_th,
      sku: product.sku,
      mainCategory: buildCategoryChain(product.category_id, nodesById)[0] ?? null,
    })),
    usedSkus.map((row) => row.sku ?? '').filter(Boolean)
  )

  return {
    ids: list.map((product) => product.id),
    rows: list.map((product) =>
      toPosExportRow(
        { ...product, sku: product.sku ?? assignedSkus.get(product.id) ?? null },
        nodesById
      )
    ),
    assignedSkus,
  }
}

/**
 * GET /api/pos-export
 * ยอดรวม + จำนวนช่องที่ยังขาด + ตัวอย่างแถวแรก ๆ ให้ดูก่อนกดโหลด
 */
export async function GET(_request: NextRequest): Promise<NextResponse> {
  return withErrorHandling(async () => {
    const { rows } = await loadExportRows()

    const total = rows.length
    const issues = countPosExportIssues(rows)
    const preview = rows.slice(0, PREVIEW_LIMIT)

    return NextResponse.json({
      success: true,
      data: {
        total,
        issues,
        preview,
      },
      total,
      issues,
      preview,
    })
  })
}

const ExportRequestSchema = z.object({
  // ปิดไว้เพื่อลองโหลดดูก่อนโดยไม่กินโควตา "ส่งออกแล้ว"
  mark_exported: z.boolean().optional().default(true),
})

/** ขนาดก้อนของ update — ทั้งตอนเขียนรหัสใหม่ และตอนตีว่าส่งออกแล้วด้วย .in() */
const SKU_UPDATE_BATCH = 25

/** ชื่อไฟล์ตามวันที่ เพื่อให้เก็บหลายรอบไว้ในโฟลเดอร์เดียวกันได้ */
function exportFileName(): string {
  const today = new Date().toISOString().slice(0, 10)
  return `pos-import-${today}.txt`
}

/**
 * POST /api/pos-export
 * สร้างไฟล์ แล้วทำเครื่องหมายว่าสินค้าชุดนี้ส่งเข้า POS ไปแล้ว
 *
 * เป็น POST ไม่ใช่ GET เพราะเขียนฐานข้อมูล และ middleware กัน non-GET ไว้อยู่แล้ว
 */
export async function POST(request: NextRequest): Promise<NextResponse> {
  return withErrorHandling(async () => {
    const raw = await request.json().catch(() => ({}))
    const { mark_exported: markExported } = ExportRequestSchema.parse(raw)

    const { ids, rows, assignedSkus } = await loadExportRows()

    if (rows.length === 0) {
      return NextResponse.json(
        { success: false, error: 'ไม่มีสินค้าที่รอส่งเข้า POS' },
        { status: 409 }
      )
    }

    const fileText = buildPosExportFile(rows)

    // เขียนฐานข้อมูลหลังสร้างไฟล์สำเร็จเท่านั้น
    // ถ้า build พังก่อนหน้านี้ ของจะยังค้างรอส่งออกอยู่ ไม่หายไปเงียบ ๆ
    if (markExported) {
      // แต่ละแถวได้คนละรหัส อัปเดตรวดเดียวด้วย .in() ไม่ได้
      // ยิงทีละก้อนเพื่อไม่ให้เปิดคอนเนกชันพร้อมกันเป็นร้อย
      //
      // ถ้าพังกลางคัน บางรายการจะมีรหัสแล้วแต่ยังไม่ถูกตีว่าส่งออก — กดใหม่ได้
      // ตัวแจกรหัสข้ามรายการที่มี sku แล้ว รหัสเดิมจึงไม่ถูกเปลี่ยน
      const entries = [...assignedSkus.entries()]
      for (let start = 0; start < entries.length; start += SKU_UPDATE_BATCH) {
        const results = await Promise.all(
          entries.slice(start, start + SKU_UPDATE_BATCH).map(([productId, sku]) =>
            supabaseAdmin.from('products').update({ sku }).eq('id', productId)
          )
        )
        const failed = results.find((result) => result.error)
        if (failed?.error) throw failed.error
      }

      // ทำเครื่องหมายเป็นขั้นสุดท้าย — ต้องแน่ใจก่อนว่าทุกแถวมีรหัสครบทุกก้อนแล้ว
      //
      // แบ่งก้อนเหมือนตอนเขียนรหัส เพราะ .in() ยัด uuid ทุกตัวลง query string
      // ของ URL — 221 ตัวก็ทะลุบัฟเฟอร์ ~8KB ของ gateway แล้วได้ 414 กลับมา
      // ผลคือรหัสถูกเขียนไปแล้วแต่ไม่มีอะไรถูกตีว่าส่งออก คิวจึงค้างวนซ้ำ
      const exportedAt = new Date().toISOString()
      for (let start = 0; start < ids.length; start += SKU_UPDATE_BATCH) {
        const { error } = await supabaseAdmin
          .from('products')
          .update({ pos_exported_at: exportedAt })
          .in('id', ids.slice(start, start + SKU_UPDATE_BATCH))

        if (error) throw error
      }
    }

    return new NextResponse(fileText, {
      status: 200,
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'Content-Disposition': `attachment; filename="${exportFileName()}"`,
        'Cache-Control': 'no-store',
      },
    })
  })
}
