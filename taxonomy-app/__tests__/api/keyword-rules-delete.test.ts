/**
 * Tests for DELETE /api/keyword-rules/[id]
 *
 * @jest-environment node
 */
import { NextRequest } from 'next/server'

let deleteOutcome = true
let deleteThrows: Error | null = null

jest.mock('@/utils/admin-db', () => ({
  deleteRow: jest.fn(async () => {
    if (deleteThrows) throw deleteThrows
    return deleteOutcome
  })
}))

import { DELETE } from '@/app/api/keyword-rules/[id]/route'

const del = (id: string) =>
  DELETE(new NextRequest(`http://127.0.0.1:3000/api/keyword-rules/${id}`, { method: 'DELETE' }), {
    params: { id }
  })

beforeEach(() => {
  deleteOutcome = true
  deleteThrows = null
})

describe('DELETE /api/keyword-rules/[id]', () => {
  it('ลบกฎสำเร็จเมื่อพบ id ในฐานข้อมูล: คืน 200', async () => {
    const res = await del('rule-uuid-1')
    const body = await res.json()

    expect(res.status).toBe(200)
    expect(body.success).toBe(true)
  })

  it('คืน 404 เมื่อไม่พบแถวที่ต้องการลบ', async () => {
    deleteOutcome = false

    const res = await del('missing-rule-uuid')
    const body = await res.json()

    expect(res.status).toBe(404)
    expect(body.success).toBe(false)
    expect(body.error).toContain('ไม่พบกฎที่ต้องการลบ')
  })

  it('คืน 500 เมื่อเกิดข้อผิดพลาดระดับฐานข้อมูล', async () => {
    deleteThrows = new Error('Database connection failure')

    const res = await del('rule-uuid-err')
    const body = await res.json()

    expect(res.status).toBe(500)
    expect(body.success).toBe(false)
  })
})
