/**
 * middleware กั้นเฉพาะ non-GET เพราะตอนออกแบบไว้ ทุกเส้นทาง GET อ่านได้แค่ข้อมูลสาธารณะ
 * แต่ตอนนี้มี route ที่อ่านผ่าน service role ข้าม RLS ไปแล้ว (`/api/settings`,
 * `/api/import/history`) — ปล่อย GET ไว้เท่ากับเปิดข้อมูลที่ตั้งใจปิดจาก anon ให้ใครก็ได้
 * ในวง LAN อ่านได้ เทสต์ชุดนี้ล็อกว่าเส้นทางกลุ่มนั้นต้องปลดล็อกก่อนเท่านั้น
 *
 * @jest-environment node
 */
import { NextRequest } from 'next/server'

jest.mock('@/utils/internal-auth', () => ({
  SESSION_COOKIE_NAME: 'internal_session',
  isValidSessionCookie: jest.fn(async (value?: string) => value === 'good-cookie')
}))

import { middleware } from '@/middleware'

const request = (path: string, init?: { method?: string; cookie?: string }) => {
  const req = new NextRequest(`http://127.0.0.1:3000${path}`, { method: init?.method ?? 'GET' })
  if (init?.cookie) req.cookies.set('internal_session', init.cookie)
  return req
}

describe('middleware', () => {
  it('ปล่อย GET ของข้อมูลทั่วไปผ่านเหมือนเดิม', async () => {
    const response = await middleware(request('/api/products'))
    expect(response.status).toBe(200)
  })

  it.each(['/api/settings', '/api/import/history'])(
    'กั้น GET %s ที่อ่านผ่าน service role ข้าม RLS',
    async (path) => {
      const response = await middleware(request(path))
      expect(response.status).toBe(401)
    }
  )

  it('ยอมให้อ่านเมื่อปลดล็อกแล้ว', async () => {
    const response = await middleware(request('/api/settings', { cookie: 'good-cookie' }))
    expect(response.status).toBe(200)
  })

  it('ยังกั้นการเขียนทุกเส้นทางเหมือนเดิม', async () => {
    const response = await middleware(request('/api/taxonomy', { method: 'POST' }))
    expect(response.status).toBe(401)
  })

  it('ปล่อย /api/unlock ผ่านเสมอ ไม่งั้นปลดล็อกไม่ได้เลย', async () => {
    const response = await middleware(request('/api/unlock', { method: 'POST' }))
    expect(response.status).toBe(200)
  })

  it('ปล่อย /api/lock ผ่านเสมอ แม้คุกกี้จะหมดอายุไปแล้ว ไม่งั้นกด "ออกจากระบบ" ไม่ได้', async () => {
    const response = await middleware(request('/api/lock', { method: 'POST' }))
    expect(response.status).toBe(200)
  })
})

/**
 * เดิมหน้าเว็บ (ต่างจาก /api/*) เข้าได้เสมอไม่ว่าจะปลดล็อกหรือยัง เพราะ matcher
 * ครอบแค่ /api/:path* — ผลคือเปิดเว็บครั้งแรกแล้วเห็นข้อมูลได้เลยโดยไม่ต้องกรอกรหัส
 * ตอนนี้เปลี่ยนให้หน้าเว็บก็ต้องปลดล็อกก่อนเหมือน API ยกเว้นหน้า /unlock เอง
 */
describe('middleware — หน้าเว็บ (ไม่ใช่ API)', () => {
  it('เด้งไป /unlock เมื่อเข้าหน้าเว็บโดยยังไม่ได้ปลดล็อก', async () => {
    const response = await middleware(request('/'))
    expect(response.status).toBe(307)
    const location = new URL(response.headers.get('location')!)
    expect(location.pathname).toBe('/unlock')
    expect(location.searchParams.get('next')).toBe('/')
  })

  it('จำหน้าที่ตั้งใจจะไปไว้ใน ?next เพื่อพากลับไปที่เดิมหลังปลดล็อก', async () => {
    const response = await middleware(request('/data-quality'))
    const location = new URL(response.headers.get('location')!)
    expect(location.searchParams.get('next')).toBe('/data-quality')
  })

  it('ปล่อยหน้าเว็บผ่านเมื่อปลดล็อกแล้ว', async () => {
    const response = await middleware(request('/', { cookie: 'good-cookie' }))
    expect(response.status).toBe(200)
  })

  it('เข้าหน้า /unlock เองได้เสมอ ไม่งั้นปลดล็อกไม่ได้เลย (วนลูปเด้งไม่รู้จบ)', async () => {
    const response = await middleware(request('/unlock'))
    expect(response.status).toBe(200)
  })
})
