import { NextRequest, NextResponse } from 'next/server'
import { z } from 'zod'
import { deleteRow, updateRow } from '@/utils/admin-db'
import { withErrorHandling } from '@/utils/error-handler'

const UUID_REGEX = /^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/
const ParamIdSchema = z.string().regex(UUID_REGEX, 'id ต้องเป็น UUID ที่ถูกต้อง')

const UpdateSynonymSchema = z.object({
  name_th: z.string().min(1).optional(),
  name: z.string().min(1).optional(),
  code: z.string().optional(),
  description: z.string().nullable().optional(),
  category_id: z.string().regex(UUID_REGEX, 'category_id ต้องเป็น UUID').nullable().optional(),
  is_active: z.boolean().optional(),
}).passthrough()

/**
 * PUT/DELETE /api/synonyms/[id]
 * เขียนผ่าน service role และตอบ 404 เมื่อไม่มีแถวถูกแตะจริง — ดูเหตุผลใน
 * `utils/admin-db.ts`
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

    const bodyValidation = UpdateSynonymSchema.safeParse(body)
    if (!bodyValidation.success) {
      return NextResponse.json(
        { success: false, error: 'ข้อมูลไม่ถูกต้อง', details: bodyValidation.error.issues },
        { status: 400 }
      )
    }

    const synonym = await updateRow('synonym_lemmas', params.id, {
      ...bodyValidation.data,
      updated_at: new Date().toISOString()
    })

    if (!synonym) {
      return NextResponse.json(
        { success: false, error: 'ไม่พบ synonym ที่ต้องการแก้ไข' },
        { status: 404 }
      )
    }

    return NextResponse.json({ success: true, data: synonym })
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

    const deleted = await deleteRow('synonym_lemmas', params.id)

    if (!deleted) {
      return NextResponse.json(
        { success: false, error: 'ไม่พบ synonym ที่ต้องการลบ' },
        { status: 404 }
      )
    }

    return NextResponse.json({ success: true })
  })
}
