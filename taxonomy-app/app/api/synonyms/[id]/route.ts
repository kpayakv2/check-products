import { NextRequest, NextResponse } from 'next/server'
import { deleteRow, updateRow } from '@/utils/admin-db'
import { withErrorHandling } from '@/utils/error-handler'

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
    const body = await request.json()
    const synonym = await updateRow('synonym_lemmas', params.id, {
      ...body,
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
