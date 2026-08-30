import { NextResponse } from 'next/server'
import { SESSION_COOKIE_NAME } from '@/utils/internal-auth'

/**
 * POST /api/lock
 * ลบคุกกี้ `internal_session` ทิ้ง — ตรงข้ามกับ /api/unlock
 * ระบบนี้ไม่มีบัญชีผู้ใช้ ("ออกจากระบบ" จึงแปลว่า "ลบคุกกี้ที่ทุกคนใช้ร่วมกัน"
 * ไม่ใช่การเพิกถอนสิทธิ์เฉพาะคนใดคนหนึ่ง — ดู CLAUDE.md → Authentication)
 */
export async function POST(): Promise<NextResponse> {
  const response = NextResponse.json({ success: true })
  response.cookies.delete(SESSION_COOKIE_NAME)
  return response
}
