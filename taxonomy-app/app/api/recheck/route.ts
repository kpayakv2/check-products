import { NextRequest, NextResponse } from 'next/server'
import { z } from 'zod'
import { supabaseAdmin } from '@/utils/supabase-admin'
import { rateLimit } from '@/utils/rate-limit'
import { withErrorHandling } from '@/utils/error-handler'
import { validateRequest } from '@/utils/validation'

const RECHECK_METHOD = 'recheck_legacy'

const DecisionSchema = z.object({
  suggestion_id: z.string().uuid('suggestion_id ต้องเป็น UUID ที่ถูกต้อง'),
  // keep = ยืนยันหมวดเดิมที่คนจัดไว้, accept = ใช้หมวดที่ AI เสนอ, override = เลือกหมวดอื่นเอง
  action: z.enum(['keep', 'accept', 'override']),
  category_id: z.string().uuid('category_id ต้องเป็น UUID ที่ถูกต้อง').optional(),
})

const limiter = rateLimit({
  interval: 60 * 1000,
  uniqueTokenPerInterval: 500,
})

/**
 * GET /api/recheck
 * ดึงรายการที่ AI ตรวจซ้ำแล้วเห็นต่างจากหมวดที่คนจัดไว้ พร้อมหมวดทั้งสองฝั่ง
 */
export async function GET(request: NextRequest): Promise<NextResponse> {
  return withErrorHandling(async () => {
    const { searchParams } = new URL(request.url)
    const limit = Math.min(Number(searchParams.get('limit')) || 25, 100)
    const offset = Math.max(Number(searchParams.get('offset')) || 0, 0)

    // นับเฉพาะรายการที่ยังไม่ถูกรีวิว เพื่อให้ badge บอกงานที่เหลือจริง
    const { count } = await supabaseAdmin
      .from('product_category_suggestions')
      .select('*', { count: 'exact', head: true })
      .eq('suggestion_method', RECHECK_METHOD)
      .eq('metadata->agrees', false)
      .is('is_accepted', null)

    const { data, error } = await supabaseAdmin
      .from('product_category_suggestions')
      .select(`
        id,
        confidence_score,
        metadata,
        suggested_category_id,
        product:products ( id, name_th, sku, category_id ),
        suggested_category:taxonomy_nodes!product_category_suggestions_suggested_category_id_fkey ( id, name_th )
      `)
      .eq('suggestion_method', RECHECK_METHOD)
      .eq('metadata->agrees', false)
      .is('is_accepted', null)
      .order('confidence_score', { ascending: false })
      .range(offset, offset + limit - 1)

    if (error) throw error

    return NextResponse.json({
      success: true,
      data: data ?? [],
      pagination: {
        total: count ?? 0,
        limit,
        offset,
        has_more: (count ?? 0) > offset + limit,
      },
    })
  })
}

/**
 * POST /api/recheck
 * บันทึกการตัดสินใจของคน — อัปเดตหมวดของสินค้าเดิม (ไม่สร้างแถวใหม่)
 * เก็บประวัติลง review_history และส่งให้ระบบเรียนรู้คีย์เวิร์ดต่อ
 */
export async function POST(request: NextRequest): Promise<NextResponse> {
  return withErrorHandling(async () => {
    try {
      await limiter.check(60, 'recheck-decision')
    } catch {
      return NextResponse.json(
        { success: false, error: 'ส่งคำขอถี่เกินไป กรุณารอสักครู่' },
        { status: 429 }
      )
    }

    const body = await validateRequest(request, DecisionSchema)
    if ('error' in body) {
      return NextResponse.json(body, { status: 400 })
    }
    const { suggestion_id, action, category_id } = body

    const { data: suggestion, error: fetchError } = await supabaseAdmin
      .from('product_category_suggestions')
      .select('id, product_id, suggested_category_id, metadata')
      .eq('id', suggestion_id)
      .single()

    if (fetchError || !suggestion) {
      return NextResponse.json(
        { success: false, error: 'ไม่พบรายการที่ต้องการ' },
        { status: 404 }
      )
    }

    const previousCategoryId = (suggestion.metadata as Record<string, unknown>)
      ?.current_category_id as string | undefined

    let finalCategoryId: string | undefined
    if (action === 'keep') finalCategoryId = previousCategoryId
    else if (action === 'accept') finalCategoryId = suggestion.suggested_category_id ?? undefined
    else finalCategoryId = category_id

    if (!finalCategoryId) {
      return NextResponse.json(
        { success: false, error: 'ต้องระบุ category_id เมื่อเลือกหมวดเอง' },
        { status: 400 }
      )
    }

    // แก้หมวดของสินค้าเดิม ไม่ใช่สร้างสินค้าใหม่
    if (finalCategoryId !== previousCategoryId) {
      const { error: updateError } = await supabaseAdmin
        .from('products')
        .update({
          category_id: finalCategoryId,
          reviewed_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        })
        .eq('id', suggestion.product_id)

      if (updateError) throw updateError

      // ตารางนี้ออกแบบไว้เก็บประวัติเปลี่ยนหมวดแต่ไม่เคยมี UI ไหนเขียนเลย
      await supabaseAdmin.from('review_history').insert({
        product_id: suggestion.product_id,
        old_category_id: previousCategoryId ?? null,
        new_category_id: finalCategoryId,
        action: `recheck_${action}`,
        comments: 'ตรวจซ้ำหมวดหมู่สินค้าเก่าผ่านหน้า Data Quality',
      })
    }

    await supabaseAdmin
      .from('product_category_suggestions')
      .update({ is_accepted: action === 'accept', reviewed_at: new Date().toISOString() })
      .eq('id', suggestion_id)

    // ให้ระบบเรียนคีย์เวิร์ดจากหมวดที่คนยืนยัน — ไม่ให้ล้มทั้งคำขอถ้า FastAPI ไม่ว่าง
    let learned = false
    try {
      const { data: product } = await supabaseAdmin
        .from('products')
        .select('name_th')
        .eq('id', suggestion.product_id)
        .single()

      if (product?.name_th) {
        const fastapiUrl = process.env.FASTAPI_URL || 'http://127.0.0.1:8000'
        const response = await fetch(`${fastapiUrl}/api/v1/learn/verify`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            product_name: product.name_th,
            category_id: finalCategoryId,
          }),
        })
        learned = response.ok
      }
    } catch (learnError) {
      console.error('Auto-learning failed (ไม่กระทบการบันทึก):', learnError)
    }

    return NextResponse.json({
      success: true,
      action,
      category_id: finalCategoryId,
      learned,
    })
  })
}
