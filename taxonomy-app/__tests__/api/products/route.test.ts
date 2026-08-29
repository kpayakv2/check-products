/**
 * เดิมไฟล์นี้ทดสอบแต่ฟังก์ชันสร้าง fixture ของตัวเอง ไม่เคยเรียก route จริงเลย
 * เขียนใหม่ให้ยิง route จริง โดยเฉพาะ status ที่ POST ใช้ตอนสร้างสินค้า —
 * ค่าเดิมคือ 'pending' ซึ่งไม่มีหน้าไหนในระบบมองเห็น
 *
 * @jest-environment node
 */
import { NextRequest } from 'next/server'

const createProduct = jest.fn(async (payload: Record<string, unknown>) => ({
  id: 'p-1',
  ...payload
}))

jest.mock('@/utils/supabase', () => {
  const actual = jest.requireActual('@/utils/product-status')
  return {
    ...actual,
    DatabaseService: {
      createProduct: (payload: Record<string, unknown>) => createProduct(payload),
      createProductAttribute: jest.fn(),
      getProducts: jest.fn(async () => [])
    }
  }
})

import { POST } from '@/app/api/products/route'
import { PENDING_REVIEW_STATUSES } from '@/utils/product-status'

const post = (body: unknown) =>
  POST(
    new NextRequest('http://127.0.0.1:3000/api/products', {
      method: 'POST',
      body: JSON.stringify(body)
    })
  )

beforeEach(() => {
  createProduct.mockClear()
})

describe('POST /api/products', () => {
  it('สร้างสินค้าด้วยสถานะที่หน้ารีวิวมองเห็นจริง', async () => {
    await post({ name_th: 'ปากกาลูกลื่น' })

    const [payload] = createProduct.mock.calls[0]
    expect(PENDING_REVIEW_STATUSES).toContain(payload.status)
  })

  it('ปฏิเสธคำขอที่ไม่มีชื่อสินค้า', async () => {
    const response = await post({ name_en: 'ball pen' })

    expect(response.status).toBe(400)
    expect(createProduct).not.toHaveBeenCalled()
  })
})
