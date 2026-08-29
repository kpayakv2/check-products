import { NextRequest, NextResponse } from 'next/server'
import { z } from 'zod'
import { supabaseAdmin } from '@/utils/supabase-admin'
import { createRow, deleteRow, readRowById, updateRow } from '@/utils/admin-db'
import { rateLimit } from '@/utils/rate-limit'
import { withErrorHandling } from '@/utils/error-handler'
import { validateRequest } from '@/utils/validation'

/**
 * POST /api/verify
 *
 * งานตรวจสองด่านในหน้า /data-quality → แท็บ Verify
 * เดิมหน้าเว็บเขียน `products` / `human_feedback` ตรงด้วย anon key ซึ่ง RLS ปิดอยู่
 * ผลคือ PATCH ได้ 200 พร้อม array ว่าง และ DELETE ได้ 204 ทั้งที่ไม่มีแถวไหนเปลี่ยน
 * ไม่มี error ให้จับ หน้าเว็บจึงรีเฟรชแล้วเจอรายการเดิมค้างอยู่ที่เดิม
 */

const DecisionSchema = z.object({
  product_id: z.string().uuid('product_id ต้องเป็น UUID ที่ถูกต้อง'),
  // keep = ไม่ซ้ำ ส่งไปด่านตรวจหมวดหมู่ต่อ, discard = ซ้ำจริง ลบทิ้ง
  action: z.enum(['keep', 'discard', 'confirm_category']),
  category_id: z.string().uuid('category_id ต้องเป็น UUID ที่ถูกต้อง').optional(),
})

const limiter = rateLimit({
  interval: 60 * 1000,
  uniqueTokenPerInterval: 500,
})

interface ProductRow {
  id: string
  name_th: string
  category_id?: string | null
  status?: string
  metadata?: Record<string, unknown> | null
}

/**
 * human_feedback เก็บ "คู่ที่คนตัดสินว่าซ้ำ/ไม่ซ้ำ" ไม่ใช่ผลจัดหมวด
 * คอลัมน์ที่โค้ดเดิมส่งไป (product_id/category_id/is_correct/comment) ไม่มีในตารางเลย
 * และ old_product/new_product/similarity_score/human_decision เป็น NOT NULL ทั้งหมด
 */
async function recordDedupFeedback(product: ProductRow, decision: 'duplicate' | 'different') {
  const metadata = (product.metadata ?? {}) as Record<string, unknown>
  await createRow('human_feedback', {
    old_product: String(metadata.duplicate_of ?? 'ไม่ระบุของเดิม'),
    new_product: product.name_th,
    similarity_score: Number(metadata.similarity_score ?? 0),
    human_decision: decision,
    comments: 'ตัดสินจากหน้า Data Quality → Verify',
  })
}

/** สอนคีย์เวิร์ดจากหมวดที่คนยืนยัน — ไม่ให้ล้มทั้งคำขอถ้า FastAPI ไม่ว่าง */
async function learnFromDecision(productName: string, categoryId: string): Promise<boolean> {
  try {
    const fastapiUrl = process.env.FASTAPI_URL || 'http://127.0.0.1:8000'
    const response = await fetch(`${fastapiUrl}/api/v1/learn/verify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ product_name: productName, category_id: categoryId }),
    })
    return response.ok
  } catch (error) {
    console.error('Auto-learning failed (ไม่กระทบการบันทึก):', error)
    return false
  }
}

export async function POST(request: NextRequest): Promise<NextResponse> {
  return withErrorHandling(async () => {
    try {
      await limiter.check(120, 'verify-decision')
    } catch {
      return NextResponse.json(
        { success: false, error: 'ส่งคำขอถี่เกินไป กรุณารอสักครู่' },
        { status: 429 }
      )
    }

    const body = await validateRequest(request, DecisionSchema)
    if ('error' in body) {
      return NextResponse.json({ success: false, ...body }, { status: 400 })
    }
    const { product_id, action, category_id } = body

    if (action === 'confirm_category' && !category_id) {
      return NextResponse.json(
        { success: false, error: 'ต้องเลือกหมวดหมู่ก่อนยืนยัน' },
        { status: 400 }
      )
    }

    const product = await readRowById<ProductRow>('products', product_id)
    if (!product) {
      return NextResponse.json(
        { success: false, error: 'ไม่พบสินค้ารายการนี้แล้ว (อาจถูกตัดสินไปก่อนหน้านี้)' },
        { status: 404 }
      )
    }

    if (action === 'discard') {
      // เก็บคำตัดสินก่อนลบ — review_history ผูก FK แบบ CASCADE จึงหายไปพร้อมสินค้า
      // ส่วน human_feedback ไม่มี FK ถึง products ประวัติจึงอยู่ต่อได้
      await recordDedupFeedback(product, 'duplicate')

      const deleted = await deleteRow('products', product_id)
      if (!deleted) {
        return NextResponse.json(
          { success: false, error: 'ลบไม่สำเร็จ ไม่มีแถวไหนถูกลบ' },
          { status: 404 }
        )
      }
      return NextResponse.json({ success: true, action })
    }

    if (action === 'keep') {
      await recordDedupFeedback(product, 'different')

      const updated = await updateRow('products', product_id, {
        status: 'pending_review_category',
        updated_at: new Date().toISOString(),
      })
      if (!updated) {
        return NextResponse.json(
          { success: false, error: 'บันทึกไม่สำเร็จ ไม่มีแถวไหนถูกแก้' },
          { status: 404 }
        )
      }
      return NextResponse.json({ success: true, action })
    }

    const finalCategoryId = category_id as string
    const updated = await updateRow('products', product_id, {
      category_id: finalCategoryId,
      status: 'approved',
      reviewed_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    })
    if (!updated) {
      return NextResponse.json(
        { success: false, error: 'บันทึกไม่สำเร็จ ไม่มีแถวไหนถูกแก้' },
        { status: 404 }
      )
    }

    // ประวัติการตรวจ — ไม่ให้ล้มทั้งคำขอถ้าเขียนประวัติไม่ผ่าน เพราะของหลักบันทึกไปแล้ว
    const { error: historyError } = await supabaseAdmin.from('review_history').insert({
      product_id,
      old_category_id: product.category_id ?? null,
      new_category_id: finalCategoryId,
      action: 'verify_category',
      comments: 'ยืนยันหมวดหมู่จากหน้า Data Quality → Verify',
    })
    if (historyError) {
      console.error('บันทึก review_history ไม่สำเร็จ:', historyError)
    }

    const learned = await learnFromDecision(product.name_th, finalCategoryId)

    return NextResponse.json({ success: true, action, category_id: finalCategoryId, learned })
  })
}
