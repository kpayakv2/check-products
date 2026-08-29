/**
 * VerifyTab เขียน `products` และ `human_feedback` ตรงด้วย anon key ซึ่ง RLS ปิดอยู่
 * ยิงจริงแล้วได้ HTTP 200 พร้อม array ว่าง — ไม่มีแถวไหนเปลี่ยน ไม่มี error ให้จับ
 * งานตรวจ 368 รายการจึงกดแล้วไม่เกิดอะไรขึ้นเลย
 *
 * อีกจุดหนึ่ง: คอลัมน์ที่โค้ดเดิมส่งให้ human_feedback (product_id/category_id/
 * is_correct/comment) ไม่มีอยู่ในตารางเลย — ต่อให้ใช้ service role ก็ยัง insert ไม่ผ่าน
 * ตารางนี้เก็บ "คู่ที่คนตัดสินว่าซ้ำ/ไม่ซ้ำ" (old_product/new_product/
 * similarity_score/human_decision) ซึ่งตรงกับด่านตรวจของซ้ำพอดี
 *
 * @jest-environment node
 */
import { NextRequest } from 'next/server'

type Filter = { kind: string; column?: string; value?: unknown }
type Call = { table: string; op: string; payload?: unknown; filters: Filter[] }

const calls: Call[] = []
let results: Record<string, { data?: unknown; error?: unknown }> = {}

const resultFor = (table: string, op: string) =>
  results[`${table}.${op}`] ?? { data: null, error: null }

const makeChain = (table: string, op: string, payload?: unknown) => {
  const call: Call = { table, op, payload, filters: [] }
  calls.push(call)
  const chain: any = {
    select: () => chain,
    eq: (column: string, value: unknown) => {
      call.filters.push({ kind: 'eq', column, value })
      return chain
    },
    is: () => chain,
    order: () => chain,
    limit: () => chain,
    single: async () => resultFor(table, op),
    maybeSingle: async () => resultFor(table, op),
    then: (resolve: (v: unknown) => unknown) => Promise.resolve(resultFor(table, op)).then(resolve)
  }
  return chain
}

jest.mock('@/utils/supabase-admin', () => ({
  supabaseAdmin: {
    from: (table: string) => ({
      select: () => makeChain(table, 'select'),
      insert: (payload: unknown) => makeChain(table, 'insert', payload),
      update: (payload: unknown) => makeChain(table, 'update', payload),
      delete: () => makeChain(table, 'delete')
    })
  }
}))

jest.mock('@/utils/rate-limit', () => ({
  rateLimit: () => ({ check: async () => undefined }),
  getClientIP: () => '127.0.0.1'
}))

import { POST as verify } from '@/app/api/verify/route'
import { DELETE as deleteKeywordRule } from '@/app/api/keyword-rules/[id]/route'

const PRODUCT_ID = '11111111-1111-4111-8111-111111111111'
const CATEGORY_ID = '22222222-2222-4222-8222-222222222222'
const RULE_ID = '33333333-3333-4333-8333-333333333333'

const post = (body: unknown) =>
  new NextRequest('http://127.0.0.1:3000/api/verify', {
    method: 'POST',
    body: JSON.stringify(body)
  })

const callsTo = (table: string, op: string) => calls.filter(c => c.table === table && c.op === op)

const productRow = {
  id: PRODUCT_ID,
  name_th: 'ปากกาลูกลื่น สีน้ำเงิน',
  category_id: CATEGORY_ID,
  status: 'pending_review_dedup',
  metadata: { duplicate_of: 'ปากกาลูกลื่น น้ำเงิน', similarity_score: 0.93 }
}

beforeEach(() => {
  calls.length = 0
  results = {}
  global.fetch = jest.fn(async () => ({ ok: true, json: async () => ({}) })) as unknown as typeof fetch
})

describe('POST /api/verify — ด่าน 1 ตรวจของซ้ำ', () => {
  it('ย้ายของที่ไม่ซ้ำไปด่านตรวจหมวดหมู่ผ่าน service role', async () => {
    results['products.select'] = { data: productRow, error: null }
    results['products.update'] = { data: { id: PRODUCT_ID }, error: null }
    results['human_feedback.insert'] = { data: { id: 'hf-1' }, error: null }

    const response = await verify(post({ product_id: PRODUCT_ID, action: 'keep' }))

    expect(response.status).toBe(200)
    const [update] = callsTo('products', 'update')
    expect((update.payload as Record<string, unknown>).status).toBe('pending_review_category')
  })

  it('บันทึกคำตัดสินลง human_feedback ด้วยคอลัมน์ที่ตารางนี้มีจริง', async () => {
    results['products.select'] = { data: productRow, error: null }
    results['products.update'] = { data: { id: PRODUCT_ID }, error: null }
    results['human_feedback.insert'] = { data: { id: 'hf-1' }, error: null }

    await verify(post({ product_id: PRODUCT_ID, action: 'keep' }))

    const [feedback] = callsTo('human_feedback', 'insert')
    const payload = feedback.payload as Record<string, unknown>
    expect(payload.human_decision).toBe('different')
    expect(payload.old_product).toBe('ปากกาลูกลื่น น้ำเงิน')
    expect(payload.new_product).toBe('ปากกาลูกลื่น สีน้ำเงิน')
    expect(payload.similarity_score).toBe(0.93)
    expect(payload.product_id).toBeUndefined()
  })

  it('เก็บคำตัดสินไว้ก่อนแล้วค่อยลบของซ้ำทิ้ง', async () => {
    results['products.select'] = { data: productRow, error: null }
    results['human_feedback.insert'] = { data: { id: 'hf-1' }, error: null }
    results['products.delete'] = { data: [{ id: PRODUCT_ID }], error: null }

    const response = await verify(post({ product_id: PRODUCT_ID, action: 'discard' }))

    expect(response.status).toBe(200)
    const feedbackIndex = calls.findIndex(c => c.table === 'human_feedback' && c.op === 'insert')
    const deleteIndex = calls.findIndex(c => c.table === 'products' && c.op === 'delete')
    expect(feedbackIndex).toBeGreaterThanOrEqual(0)
    expect(feedbackIndex).toBeLessThan(deleteIndex)
    expect((calls[feedbackIndex].payload as Record<string, unknown>).human_decision).toBe('duplicate')
  })

  it('ตอบ 404 เมื่อไม่มีสินค้านั้นแล้ว แทนที่จะรายงานว่าสำเร็จ', async () => {
    results['products.select'] = { data: null, error: null }

    const response = await verify(post({ product_id: PRODUCT_ID, action: 'keep' }))

    expect(response.status).toBe(404)
    expect(callsTo('products', 'update')).toHaveLength(0)
  })
})

describe('POST /api/verify — ด่าน 2 ยืนยันหมวดหมู่', () => {
  it('ไม่ยอมบันทึกเมื่อยังไม่ได้เลือกหมวดหมู่', async () => {
    const response = await verify(post({ product_id: PRODUCT_ID, action: 'confirm_category' }))

    expect(response.status).toBe(400)
    expect(callsTo('products', 'update')).toHaveLength(0)
  })

  it('อนุมัติสินค้าพร้อมหมวดที่คนเลือก และเก็บประวัติลง review_history', async () => {
    results['products.select'] = { data: { ...productRow, status: 'pending_review_category' }, error: null }
    results['products.update'] = { data: { id: PRODUCT_ID }, error: null }
    results['review_history.insert'] = { data: [{ id: 'rh-1' }], error: null }

    const response = await verify(
      post({ product_id: PRODUCT_ID, action: 'confirm_category', category_id: CATEGORY_ID })
    )

    expect(response.status).toBe(200)
    const [update] = callsTo('products', 'update')
    const payload = update.payload as Record<string, unknown>
    expect(payload.status).toBe('approved')
    expect(payload.category_id).toBe(CATEGORY_ID)
    expect(callsTo('review_history', 'insert')).toHaveLength(1)
  })

  it('ตอบ 404 เมื่อไม่มีแถวไหนถูกแก้จริง', async () => {
    results['products.select'] = { data: { ...productRow, status: 'pending_review_category' }, error: null }
    results['products.update'] = { data: null, error: null }

    const response = await verify(
      post({ product_id: PRODUCT_ID, action: 'confirm_category', category_id: CATEGORY_ID })
    )

    expect(response.status).toBe(404)
  })
})

describe('DELETE /api/keyword-rules/[id]', () => {
  const del = () =>
    deleteKeywordRule(
      new NextRequest(`http://127.0.0.1:3000/api/keyword-rules/${RULE_ID}`, { method: 'DELETE' }),
      { params: { id: RULE_ID } }
    )

  it('ตอบ 404 เมื่อไม่มีกฎไหนถูกลบจริง (anon key เคยได้ 204 ทั้งที่ไม่ได้ลบ)', async () => {
    results['keyword_rules.delete'] = { data: [], error: null }

    expect((await del()).status).toBe(404)
  })

  it('ตอบสำเร็จเมื่อลบได้จริง', async () => {
    results['keyword_rules.delete'] = { data: [{ id: RULE_ID }], error: null }

    expect((await del()).status).toBe(200)
  })
})
