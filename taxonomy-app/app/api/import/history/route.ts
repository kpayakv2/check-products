import { NextRequest, NextResponse } from 'next/server'
import { supabaseAdmin } from '@/utils/supabase-admin'
import { withErrorHandling } from '@/utils/error-handler'

/**
 * GET /api/import/history
 * ประวัติการนำเข้าล่าสุด
 *
 * ต้องอ่านผ่าน service role เพราะ migration 20260828000000 จำกัด SELECT ของ
 * ตาราง `imports` ไว้ที่ role editor/admin ส่วนหน้าเว็บคุยกับ Supabase ด้วย
 * anon key — อ่านตรงๆ จากเบราว์เซอร์จะได้ผลลัพธ์ว่างเปล่าโดยไม่มี error
 */
export async function GET(request: NextRequest): Promise<NextResponse> {
  return withErrorHandling(async () => {
    const { searchParams } = new URL(request.url)
    const limit = Math.min(Number(searchParams.get('limit')) || 20, 100)

    const { data, error } = await supabaseAdmin
      .from('imports')
      .select('*')
      .order('created_at', { ascending: false })
      .limit(limit)

    if (error) throw error

    return NextResponse.json({ success: true, data: data ?? [] })
  })
}
