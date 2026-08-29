/**
 * ImportHistory เคยอ่านตาราง `imports` ตรงๆ ด้วย anon key แต่ migration
 * 20260828000000 จำกัด SELECT ของตารางนี้ไว้ที่ role editor/admin
 * ผลคือหน้าประวัติขึ้น "ยังไม่มีประวัติการนำเข้า" ตลอด ทั้งที่ในฐานข้อมูลมีข้อมูลอยู่
 * (ยืนยันด้วยเบราว์เซอร์จริง: ไม่มี error ใน console เลย มันเงียบสนิท)
 *
 * @jest-environment node
 */
import { NextRequest } from 'next/server'

const rows = [
  { id: 'i-1', name: 'ไฟล์ล่าสุด', file_name: 'a.csv', total_records: 405, created_at: '2026-08-28T00:00:00Z' }
]

let selectError: unknown = null

jest.mock('@/utils/supabase-admin', () => ({
  supabaseAdmin: {
    from: () => {
      const chain: any = {
        select: () => chain,
        order: () => chain,
        limit: async () => ({ data: rows, error: selectError }),
        then: (resolve: (v: unknown) => unknown) =>
          Promise.resolve({ data: rows, error: selectError }).then(resolve)
      }
      return chain
    }
  }
}))

jest.mock('@/utils/rate-limit', () => ({
  rateLimit: () => ({ check: async () => undefined })
}))

import { GET } from '@/app/api/import/history/route'

const get = () => GET(new NextRequest('http://127.0.0.1:3000/api/import/history'))

beforeEach(() => {
  selectError = null
})

describe('GET /api/import/history', () => {
  it('คืนประวัติการนำเข้าผ่าน service role ไม่ให้ RLS กรองทิ้งเงียบๆ', async () => {
    const response = await get()
    const body = await response.json()

    expect(response.status).toBe(200)
    expect(body.data).toHaveLength(1)
    expect(body.data[0].file_name).toBe('a.csv')
  })

  it('บอกให้รู้เมื่ออ่านฐานข้อมูลไม่สำเร็จ ไม่ใช่คืนรายการว่าง', async () => {
    selectError = { message: 'boom' }

    const response = await get()

    expect(response.status).toBeGreaterThanOrEqual(500)
  })
})
