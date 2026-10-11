import { NextRequest, NextResponse } from 'next/server'
import { z } from 'zod'
import { readSettingsRow, writeSettingsRow } from '@/utils/admin-db'
import { withErrorHandling } from '@/utils/error-handler'

const UpdateSettingsSchema = z.record(z.string(), z.unknown())

/**
 * GET/PUT /api/settings
 *
 * `system_settings` เปิดให้เฉพาะ role taxonomy_editor/admin ทั้งอ่านและเขียน
 * (migration 20260828000000) หน้า /settings ที่เรียก Supabase ตรงด้วย anon key
 * จึงได้ 406 ตอนโหลดและบันทึกไม่ลงเลย — งานทั้งสองทางต้องผ่าน service role
 */
export async function GET(): Promise<NextResponse> {
  return withErrorHandling(async () => {
    const settings = await readSettingsRow()
    return NextResponse.json({ success: true, data: settings })
  })
}

export async function PUT(request: NextRequest): Promise<NextResponse> {
  return withErrorHandling(async () => {
    let body: unknown
    try {
      body = await request.json()
    } catch {
      return NextResponse.json(
        { success: false, error: 'Invalid JSON body' },
        { status: 400 }
      )
    }

    const validation = UpdateSettingsSchema.safeParse(body)
    if (!validation.success || typeof body !== 'object' || body === null || Array.isArray(body)) {
      return NextResponse.json(
        { success: false, error: 'ข้อมูลการตั้งค่าต้องเป็น object' },
        { status: 400 }
      )
    }

    // ตัด id ที่หน้าเว็บส่งกลับมาทิ้ง — แถวเป้าหมายตัดสินจากฝั่ง server เท่านั้น
    const { id: _ignored, ...settings } = validation.data
    const saved = await writeSettingsRow(settings)
    return NextResponse.json({ success: true, data: saved })
  })
}
