import { NextRequest, NextResponse } from 'next/server'
import { deleteRow } from '@/utils/admin-db'
import { withErrorHandling } from '@/utils/error-handler'

/**
 * DELETE /api/keyword-rules/[id]
 *
 * แท็บ Auto-learn เคยลบกฎด้วย anon key ซึ่ง RLS กรองแถวทิ้งก่อน — ได้ 204 กลับมา
 * โดยไม่มีแถวไหนถูกลบ แล้วหน้าเว็บก็ตัดการ์ดออกจากจอทันที ดูเหมือนลบสำเร็จ
 * จนกว่าจะกดรีเฟรชแล้วเจอกฎเดิมกลับมา
 */
export async function DELETE(
  _request: NextRequest,
  { params }: { params: { id: string } }
): Promise<NextResponse> {
  return withErrorHandling(async () => {
    const deleted = await deleteRow('keyword_rules', params.id)

    if (!deleted) {
      return NextResponse.json(
        { success: false, error: 'ไม่พบกฎที่ต้องการลบ' },
        { status: 404 }
      )
    }

    return NextResponse.json({ success: true })
  })
}
