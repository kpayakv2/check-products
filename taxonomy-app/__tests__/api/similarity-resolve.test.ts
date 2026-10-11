/**
 * Tests for POST /api/similarity/resolve
 *
 * @jest-environment node
 */
import { NextRequest } from 'next/server'

let updateResult = { data: [{ id: '22222222-2222-2222-2222-222222222222' }], error: null as any }
let upsertResult = { data: null, error: null as any }

jest.mock('@/utils/supabase-admin', () => ({
  supabaseAdmin: {
    from: (table: string) => {
      const chain: any = {
        update: () => chain,
        eq: () => chain,
        select: async () => updateResult,
        upsert: async () => upsertResult
      }
      return chain
    }
  }
}))

jest.mock('@/utils/rate-limit', () => ({
  rateLimit: () => ({ check: async () => undefined })
}))

import { POST } from '@/app/api/similarity/resolve/route'

const post = (body: Record<string, unknown>) =>
  POST(
    new NextRequest('http://127.0.0.1:3000/api/similarity/resolve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    })
  )

beforeEach(() => {
  updateResult = { data: [{ id: '22222222-2222-2222-2222-222222222222' }], error: null }
  upsertResult = { data: null, error: null }
})

describe('POST /api/similarity/resolve', () => {
  const idA = '11111111-1111-1111-1111-111111111111'
  const idB = '22222222-2222-2222-2222-222222222222'

  it('merge สำเร็จ: อัปเดตสถานะ duplicate และบันทึก similarity_matches', async () => {
    const res = await post({
      action: 'merge',
      id_a: idA,
      id_b: idB,
      keep_id: idA,
      similarity: 0.95
    })
    const body = await res.json()

    expect(res.status).toBe(200)
    expect(body.success).toBe(true)
    expect(body.message).toContain('successfully merged')
  })

  it('merge ล้มเหลวเมื่อไม่ระบุ keep_id: คืน 400', async () => {
    const res = await post({
      action: 'merge',
      id_a: idA,
      id_b: idB,
      similarity: 0.95
    })
    const body = await res.json()

    expect(res.status).toBe(400)
    expect(body.success).toBe(false)
    expect(body.error).toContain('keep_id is required')
  })

  it('merge ล้มเหลวเมื่อไม่พบสินค้าที่ต้องการลบ: คืน 404', async () => {
    updateResult = { data: [], error: null }

    const res = await post({
      action: 'merge',
      id_a: idA,
      id_b: idB,
      keep_id: idA,
      similarity: 0.95
    })
    const body = await res.json()

    expect(res.status).toBe(404)
    expect(body.success).toBe(false)
    expect(body.error).toContain('ไม่พบสินค้าที่ต้องการลบ')
  })

  it('keep_both สำเร็จ: บันทึกยกเว้นใน similarity_matches', async () => {
    const res = await post({
      action: 'keep_both',
      id_a: idA,
      id_b: idB,
      similarity: 0.85
    })
    const body = await res.json()

    expect(res.status).toBe(200)
    expect(body.success).toBe(true)
    expect(body.message).toContain('Decided to keep both')
  })

  it('reject request เมื่อ input ไม่ถูกต้อง (invalid UUID)', async () => {
    const res = await post({
      action: 'merge',
      id_a: 'not-a-uuid',
      id_b: idB,
      keep_id: idA
    })

    expect(res.status).toBe(400)
  })
})
