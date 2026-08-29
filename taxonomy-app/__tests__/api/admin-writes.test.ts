/**
 * ทุกหน้าในแอปคุยกับ Supabase ด้วย anon key แต่ policy ของ taxonomy_nodes /
 * synonym_lemmas / system_settings ให้เขียนได้เฉพาะ role taxonomy_editor|admin
 * ผลที่พิสูจน์กับ DB จริงแล้ว: insert ได้ 42501, update ได้ 200 พร้อม array ว่าง
 * (ไม่มี error) และ delete ได้ 204 ทั้งที่ไม่มีแถวไหนถูกลบ — หน้าเว็บจึงขึ้นว่า
 * "ลบเรียบร้อยแล้ว" โดยที่ข้อมูลยังอยู่
 *
 * เทสต์ชุดนี้ล็อกว่า route ที่รับงานเขียนต้องใช้ service role และต้องบอกได้ว่า
 * "ไม่พบแถวที่จะแก้/ลบ" แทนที่จะเงียบแล้วรายงานว่าสำเร็จ
 *
 * @jest-environment node
 */
import { NextRequest } from 'next/server'

type Call = { table: string; op: string; payload?: unknown }

const calls: Call[] = []
/** ผลลัพธ์ต่อ `<table>.<op>` กำหนดจากในเทสต์ */
let results: Record<string, { data?: unknown; error?: unknown }> = {}

const resultFor = (table: string, op: string) =>
  results[`${table}.${op}`] ?? { data: null, error: null }

const makeChain = (table: string, op: string, payload?: unknown) => {
  calls.push({ table, op, payload })
  const chain: any = {
    select: () => chain,
    eq: () => chain,
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
      upsert: (payload: unknown) => makeChain(table, 'upsert', payload),
      delete: () => makeChain(table, 'delete')
    })
  }
}))

jest.mock('@/utils/rate-limit', () => ({
  rateLimit: () => ({ check: async () => undefined }),
  getClientIP: () => '127.0.0.1'
}))

import { POST as createTaxonomy } from '@/app/api/taxonomy/route'
import { PUT as updateTaxonomy, DELETE as deleteTaxonomy } from '@/app/api/taxonomy/[id]/route'
import { POST as createSynonym } from '@/app/api/synonyms/route'
import { GET as readSettings, PUT as writeSettings } from '@/app/api/settings/route'

const NODE_ID = '11111111-1111-1111-1111-111111111111'

const post = (url: string, body: unknown) =>
  new NextRequest(`http://127.0.0.1:3000${url}`, { method: 'POST', body: JSON.stringify(body) })

const put = (url: string, body: unknown) =>
  new NextRequest(`http://127.0.0.1:3000${url}`, { method: 'PUT', body: JSON.stringify(body) })

const callsTo = (table: string, op: string) => calls.filter(c => c.table === table && c.op === op)

beforeEach(() => {
  calls.length = 0
  results = {}
})

describe('POST /api/taxonomy', () => {
  it('สร้างหมวดหมู่ผ่าน service role ไม่ใช่ anon key ที่ RLS ปิดอยู่', async () => {
    results['taxonomy_nodes.insert'] = { data: { id: NODE_ID, name_th: 'เครื่องเขียน' }, error: null }

    const response = await createTaxonomy(post('/api/taxonomy', { name_th: 'เครื่องเขียน' }))

    expect(response.status).toBe(201)
    expect(callsTo('taxonomy_nodes', 'insert')).toHaveLength(1)
  })

  it('ไม่ส่ง parent_id เป็นสตริงว่างไปให้คอลัมน์ uuid', async () => {
    results['taxonomy_nodes.insert'] = { data: { id: NODE_ID }, error: null }

    await createTaxonomy(post('/api/taxonomy', { name_th: 'เครื่องเขียน', parent_id: '' }))

    const [insert] = callsTo('taxonomy_nodes', 'insert')
    expect((insert.payload as Record<string, unknown>).parent_id).toBeUndefined()
  })
})

describe('PUT /api/taxonomy/[id]', () => {
  it('บอกว่าไม่พบแถว แทนที่จะรายงานว่าแก้สำเร็จทั้งที่ไม่มีอะไรเปลี่ยน', async () => {
    results['taxonomy_nodes.update'] = { data: null, error: null }

    const response = await updateTaxonomy(put(`/api/taxonomy/${NODE_ID}`, { name_th: 'ใหม่' }), {
      params: { id: NODE_ID }
    })

    expect(response.status).toBe(404)
  })
})

describe('DELETE /api/taxonomy/[id]', () => {
  it('ตอบ 404 เมื่อไม่มีแถวไหนถูกลบจริง', async () => {
    results['taxonomy_nodes.delete'] = { data: [], error: null }

    const response = await deleteTaxonomy(
      new NextRequest(`http://127.0.0.1:3000/api/taxonomy/${NODE_ID}`, { method: 'DELETE' }),
      { params: { id: NODE_ID } }
    )

    expect(response.status).toBe(404)
  })

  it('ตอบสำเร็จเมื่อลบได้จริง', async () => {
    results['taxonomy_nodes.delete'] = { data: [{ id: NODE_ID }], error: null }

    const response = await deleteTaxonomy(
      new NextRequest(`http://127.0.0.1:3000/api/taxonomy/${NODE_ID}`, { method: 'DELETE' }),
      { params: { id: NODE_ID } }
    )

    expect(response.status).toBe(200)
  })
})

describe('POST /api/synonyms', () => {
  it('บันทึกคำพ้องที่ส่งมาด้วย ไม่ใช่ตรวจว่ามีแล้วทิ้ง', async () => {
    results['synonym_lemmas.insert'] = { data: { id: 'syn-1' }, error: null }
    results['synonym_terms.insert'] = { data: [{ id: 'term-1' }], error: null }

    const response = await createSynonym(
      post('/api/synonyms', {
        name: 'ปากกา',
        terms: [{ term: 'ปากกาลูกลื่น', is_primary: true }]
      })
    )

    expect(response.status).toBe(201)
    expect(callsTo('synonym_lemmas', 'insert')).toHaveLength(1)
    expect(callsTo('synonym_terms', 'insert')).toHaveLength(1)
  })
})

describe('/api/settings', () => {
  it('อ่านค่าตั้งค่าผ่าน service role (anon อ่านตารางนี้ไม่ได้)', async () => {
    results['system_settings.select'] = { data: { id: 's-1', ai_provider: 'local' }, error: null }

    const response = await readSettings()
    const body = await response.json()

    expect(response.status).toBe(200)
    expect(body.data.ai_provider).toBe('local')
  })

  it('บันทึกค่าตั้งค่าผ่าน service role', async () => {
    results['system_settings.upsert'] = { data: { id: 's-1', ai_provider: 'openai' }, error: null }

    const response = await writeSettings(put('/api/settings', { ai_provider: 'openai' }))

    expect(response.status).toBe(200)
    expect(callsTo('system_settings', 'upsert')).toHaveLength(1)
  })
})
