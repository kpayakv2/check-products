/**
 * route นี้ใช้ NextRequest ซึ่งต่อยอดจาก Request ของ Fetch API
 * jsdom ที่ตั้งไว้เป็นค่าเริ่มต้นไม่มี global พวกนี้ จึงต้องรันบน node
 *
 * @jest-environment node
 */
import { NextRequest } from 'next/server'

/**
 * ตัวแทน supabase client แบบบางที่สุดที่รองรับเฉพาะลูกโซ่ที่ route นี้เรียกใช้จริง
 * ผลลัพธ์ของแต่ละตาราง/แต่ละคำสั่งกำหนดจากภายนอกได้ และบันทึกไว้ว่าถูกเรียกอะไรบ้าง
 */
type Result = { data?: unknown; error?: unknown }

const calls: { table: string; op: string; payload?: unknown }[] = []
let results: Record<string, Result> = {}

const makeChain = (table: string, op: string, payload?: unknown) => {
  calls.push({ table, op, payload })
  const result = () => results[`${table}.${op}`] ?? { data: null, error: null }
  const chain: any = {
    select: () => chain,
    eq: () => chain,
    in: () => chain,
    maybeSingle: async () => result(),
    single: async () => result(),
    then: (resolve: (v: Result) => unknown) => Promise.resolve(result()).then(resolve)
  }
  return chain
}

jest.mock('@/utils/supabase-admin', () => ({
  supabaseAdmin: {
    from: (table: string) => ({
      select: (_cols?: string) => makeChain(table, 'select'),
      insert: (payload: unknown) => makeChain(table, 'insert', payload),
      update: (payload: unknown) => makeChain(table, 'update', payload),
      delete: () => makeChain(table, 'delete')
    })
  }
}))

jest.mock('@/utils/rate-limit', () => ({
  rateLimit: () => ({ check: async () => undefined })
}))

import { POST } from '@/app/api/import/commit/route'

const RUN_ID = 'run-1234567890'
const EXISTING_BATCH = '44444444-4444-4444-4444-444444444444'

const dedupBody = {
  action: 'dedup',
  wizard_run_id: RUN_ID,
  file_name: 'test.csv',
  items: [{ name_th: 'ปากกาลูกลื่น', cleaned_name: 'ปากกาลูกลื่น', bucket: 'new', similarity: 0 }]
}

const postDedup = () =>
  POST(new NextRequest('http://127.0.0.1:3000/api/import/commit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(dedupBody)
  }))

const importInserts = () => calls.filter((c) => c.table === 'imports' && c.op === 'insert')

describe('/api/import/commit — กันบันทึกซ้ำด้วย wizard_run_id', () => {
  beforeEach(() => {
    calls.length = 0
    results = {}
    // embedAll ยิงไปที่ FastAPI — ตอบให้สำเร็จไว้ ไม่ใช่ประเด็นของเทสต์ชุดนี้
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ embeddings: [[0.1, 0.2]] })
    }) as unknown as typeof fetch
  })

  it('รอบที่เคยบันทึกแล้ว คืนผลของ batch เดิมโดยไม่สร้างใหม่', async () => {
    results['imports.select'] = { data: { id: EXISTING_BATCH }, error: null }
    results['products.select'] = {
      data: [{ id: 'p1', name_th: 'ปากกาลูกลื่น', status: 'pending_review_category' }],
      error: null
    }

    const response = await postDedup()
    const payload = await (response as Response).json()

    expect(payload.success).toBe(true)
    expect(payload.reused).toBe(true)
    expect(payload.import_batch_id).toBe(EXISTING_BATCH)
    // คีย์ products ต้องมาครบ ขั้นจัดหมวดหมู่ใช้ทำ map ชื่อ → id
    expect(payload.products).toEqual([
      { id: 'p1', name_th: 'ปากกาลูกลื่น', status: 'pending_review_category' }
    ])
    expect(importInserts()).toHaveLength(0)
  })

  it('ไม่เคยบันทึก จะสร้าง batch ใหม่พร้อมแนบ wizard_run_id ตั้งแต่ตอน insert', async () => {
    results['imports.select'] = { data: null, error: null }
    results['imports.insert'] = { data: { id: 'batch-ใหม่' }, error: null }
    results['products.insert'] = {
      data: [{ id: 'p1', name_th: 'ปากกาลูกลื่น', status: 'pending_review_category' }],
      error: null
    }

    const response = await postDedup()
    const payload = await (response as Response).json()

    expect(payload.success).toBe(true)
    expect(payload.reused).toBeUndefined()

    // ต้องอยู่ใน insert ไม่ใช่ไปเติมทีหลัง ไม่งั้นช่วงขอ embedding จะเป็นช่องให้บันทึกซ้ำ
    const inserted = importInserts()[0].payload as { metadata?: { wizard_run_id?: string } }
    expect(inserted.metadata?.wizard_run_id).toBe(RUN_ID)
  })

  it('อัปเดตสถานะตอนจบต้องไม่ลบ wizard_run_id ทิ้ง', async () => {
    results['imports.select'] = { data: null, error: null }
    results['imports.insert'] = { data: { id: 'batch-ใหม่' }, error: null }
    results['products.insert'] = {
      data: [{ id: 'p1', name_th: 'ปากกาลูกลื่น', status: 'pending_review_category' }],
      error: null
    }

    await postDedup()

    const update = calls.find((c) => c.table === 'imports' && c.op === 'update')
    const metadata = (update?.payload as { metadata?: Record<string, unknown> })?.metadata
    expect(metadata?.wizard_run_id).toBe(RUN_ID)
    expect(metadata).toHaveProperty('counts')
  })

  it('สองคำขอชนกันจน unique index ตัดออก คืนผลของ batch ที่บันทึกสำเร็จแทนการโยน error', async () => {
    results['imports.select'] = { data: { id: EXISTING_BATCH }, error: null }
    results['imports.insert'] = { data: null, error: { code: '23505', message: 'duplicate key' } }
    results['products.select'] = {
      data: [{ id: 'p1', name_th: 'ปากกาลูกลื่น', status: 'pending_review_category' }],
      error: null
    }

    // ครั้งแรกต้องผ่านด่านเช็คก่อนไปได้ จึงให้ select คืน null รอบแรกแล้วค่อยเจอทีหลัง
    let firstLookup = true
    results['imports.select'] = { data: null, error: null }
    const original = results
    Object.defineProperty(original, 'imports.select', {
      get() {
        if (firstLookup) {
          firstLookup = false
          return { data: null, error: null }
        }
        return { data: { id: EXISTING_BATCH }, error: null }
      },
      configurable: true
    })

    const response = await postDedup()
    const payload = await (response as Response).json()

    expect(payload.success).toBe(true)
    expect(payload.reused).toBe(true)
    expect(payload.import_batch_id).toBe(EXISTING_BATCH)
  })

  it('ปฏิเสธคำขอที่ไม่มี wizard_run_id', async () => {
    const response = await POST(new NextRequest('http://127.0.0.1:3000/api/import/commit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...dedupBody, wizard_run_id: undefined })
    }))

    expect((response as Response).status).toBe(400)
    expect(importInserts()).toHaveLength(0)
  })
})
