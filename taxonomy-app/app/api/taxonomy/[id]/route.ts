import { NextRequest, NextResponse } from 'next/server'
import { deleteRow, updateRow } from '@/utils/admin-db'
import { withErrorHandling } from '@/utils/error-handler'

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
    const body = await request.json()
    const category = await updateRow('taxonomy_nodes', params.id, {
      ...body,
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
