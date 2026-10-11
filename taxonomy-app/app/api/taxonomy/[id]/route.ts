import { NextRequest, NextResponse } from 'next/server'
import { z } from 'zod'
import { deleteRow, updateRow } from '@/utils/admin-db'
import { withErrorHandling } from '@/utils/error-handler'

const UUID_REGEX = /^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/
const ParamIdSchema = z.string().regex(UUID_REGEX, 'id ต้องเป็น UUID ที่ถูกต้อง')

const UpdateTaxonomySchema = z.object({
  name_th: z.string().min(1).optional(),
  name_en: z.string().nullable().optional(),
  description: z.string().nullable().optional(),
  parent_id: z.preprocess(
    (val) => (val === '' ? null : val),
    z.string().regex(UUID_REGEX, 'parent_id ต้องเป็น UUID').nullable().optional()
  ),
  keywords: z.array(z.string()).optional(),
  is_active: z.boolean().optional(),
  sort_order: z.number().optional(),
}).passthrough()

/**
 * PUT/DELETE /api/taxonomy/[id]
 *
 * เขียนผ่าน service role เพราะ policy ของ `taxonomy_nodes` ให้เฉพาะ
 * taxonomy_editor/admin แก้ได้ ส่วน anon จะได้ "สำเร็จ" แบบไม่มีอะไรเปลี่ยน
 * และตอบ 404 เมื่อไม่มีแถวถูกแก้/ถูกลบจริง แทนที่จะรายงานว่าสำเร็จเปล่า ๆ
 */
export async function PUT(
  request: NextRequest,
  { params }: { params: { id: string } }
): Promise<NextResponse> {
  return withErrorHandling(async () => {
    const idValidation = ParamIdSchema.safeParse(params.id)
    if (!idValidation.success) {
      return NextResponse.json(
        { success: false, error: 'id ต้องเป็น UUID ที่ถูกต้อง' },
        { status: 400 }
      )
    }

    let body: unknown
    try {
      body = await request.json()
    } catch {
      return NextResponse.json(
        { success: false, error: 'Invalid JSON body' },
        { status: 400 }
      )
    }

    const bodyValidation = UpdateTaxonomySchema.safeParse(body)
    if (!bodyValidation.success) {
      return NextResponse.json(
        { success: false, error: 'ข้อมูลไม่ถูกต้อง', details: bodyValidation.error.issues },
        { status: 400 }
      )
    }

    const category = await updateRow('taxonomy_nodes', params.id, {
      ...bodyValidation.data,
      updated_at: new Date().toISOString()
    })

    if (!category) {
      return NextResponse.json(
        { success: false, error: 'ไม่พบหมวดหมู่ที่ต้องการแก้ไข' },
        { status: 404 }
      )
    }

    return NextResponse.json({ success: true, data: category })
  })
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: { id: string } }
): Promise<NextResponse> {
  return withErrorHandling(async () => {
    const idValidation = ParamIdSchema.safeParse(params.id)
    if (!idValidation.success) {
      return NextResponse.json(
        { success: false, error: 'id ต้องเป็น UUID ที่ถูกต้อง' },
        { status: 400 }
      )
    }

    const deleted = await deleteRow('taxonomy_nodes', params.id)

    if (!deleted) {
      return NextResponse.json(
        { success: false, error: 'ไม่พบหมวดหมู่ที่ต้องการลบ' },
        { status: 404 }
      )
    }

    return NextResponse.json({ success: true })
  })
}
