/**
 * ปุ่ม "ออกจากระบบ" ใน Header.tsx เคยไม่มี onClick เลย — กดแล้วไม่เกิดอะไรขึ้น
 * เพราะระบบนี้ไม่มีบัญชีผู้ใช้จริง มีแค่คุกกี้ `internal_session` ตัวเดียวที่ทุกคนใช้ร่วมกัน
 * (ดู CLAUDE.md → Authentication) สิ่งเดียวที่ "ออกจากระบบ" แปลว่าได้จริงคือลบคุกกี้นั้นทิ้ง
 *
 * @jest-environment node
 */
import { NextRequest } from 'next/server'
import { SESSION_COOKIE_NAME } from '@/utils/internal-auth'
import { POST } from '@/app/api/lock/route'

const request = () => new NextRequest('http://127.0.0.1:3000/api/lock', { method: 'POST' })

describe('POST /api/lock', () => {
  it('ลบคุกกี้เซสชันทิ้ง', async () => {
    const response = await POST()

    const setCookie = response.headers.get('set-cookie') ?? ''
    expect(setCookie).toContain(`${SESSION_COOKIE_NAME}=;`)
    // ลบคุกกี้ = ตั้งวันหมดอายุไว้ในอดีต ไม่ใช่แค่ส่งค่าว่าง เบราว์เซอร์ถึงจะลบออกจริง
    expect(setCookie).toMatch(/Expires=Thu, 01 Jan 1970/i)
  })

  it('ตอบ success เสมอ แม้เรียกตอนไม่มีคุกกี้อยู่แล้ว', async () => {
    const response = await POST()
    const body = await response.json()

    expect(response.status).toBe(200)
    expect(body.success).toBe(true)
  })
})
