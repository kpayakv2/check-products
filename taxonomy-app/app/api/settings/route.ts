import { NextRequest, NextResponse } from 'next/server'
import { readSingleRow, upsertRow } from '@/utils/admin-db'
import { withErrorHandling } from '@/utils/error-handler'

/**
 * GET/PUT /api/settings
 *
 * `system_settings` เปิดให้เฉพาะ role taxonomy_editor/admin ทั้งอ่านและเขียน
 * (migration 20260828000000) หน้า /settings ที่เรียก Supabase ตรงด้วย anon key
 * จึงได้ 406 ตอนโหลดและบันทึกไม่ลงเลย — งานทั้งสองทางต้องผ่าน service role
 */
export async function GET(): Promise<NextResponse> {
  return withErrorHandling(async () => {
    const settings = await readSingleRow('system_settings')
    return NextResponse.json({ success: true, data: settings })
  })
}

export async function PUT(request: NextRequest): Promise<NextResponse> {
  return withErrorHandling(async () => {
    const body = await request.json()
    const settings = await upsertRow('system_settings', body)
    return NextResponse.json({ success: true, data: settings })
  })
}
