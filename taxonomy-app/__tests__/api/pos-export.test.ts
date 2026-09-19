/**
 * route นี้อ่านด้วย service role (ข้าม RLS) จึงต้องอยู่ใน GATED_READ_PATHS
 * ของ middleware ไม่งั้น GET จะเปิดให้อ่านได้โดยไม่ต้องปลดล็อก
 *
 * @jest-environment node
 */
import { NextRequest } from 'next/server'
import { readFileSync } from 'fs'
import path from 'path'

type Filter = { kind: string; column?: string; value?: unknown }
type Call = { table: string; op: string; payload?: unknown; filters: Filter[] }

const calls: Call[] = []

/**
 * คิวผลลัพธ์ต่อ `table.op` เพราะ route ยิง products สองครั้ง:
 * ครั้งแรกเอารายการที่จะส่งออก ครั้งที่สองเอา sku ทั้งหมดไปคำนวณรหัสใหม่
 */
let queues: Record<string, Array<{ data?: unknown; error?: unknown }>> = {}

const resultFor = (table: string, op: string) =>
  queues[`${table}.${op}`]?.shift() ?? { data: [], error: null }

const makeChain = (table: string, op: string, payload?: unknown) => {
  const call: Call = { table, op, payload, filters: [] }
  calls.push(call)
  const settled = resultFor(table, op)
  const chain: Record<string, unknown> = {}
  const self = () => chain
  Object.assign(chain, {
    select: self,
    order: self,
    not: self,
    eq: (column: string, value: unknown) => {
      call.filters.push({ kind: 'eq', column, value })
      return chain
    },
    is: (column: string, value: unknown) => {
      call.filters.push({ kind: 'is', column, value })
      return chain
    },
    in: (column: string, value: unknown) => {
      call.filters.push({ kind: 'in', column, value })
      return Promise.resolve(settled)
    },
    then: (resolve: (v: unknown) => unknown) => Promise.resolve(settled).then(resolve),
  })
  return chain
}

jest.mock('@/utils/supabase-admin', () => ({
  supabaseAdmin: {
    from: (table: string) => ({
      select: (...args: unknown[]) => makeChain(table, 'select', args),
      update: (payload: unknown) => makeChain(table, 'update', payload),
    }),
  },
}))

const PRODUCTS = [
  {
    id: 'p1',
    name_th: 'ตะกร้ากลม 9899',
    sku: 'HH001',
    barcode: null,
    unit_name: 'ใบ',
    vat_flag: 'V',
    price: 45,
    category_id: 'sub',
  },
  {
    id: 'p2',
    name_th: 'เทปปะหม้อม้วนเล็ก',
    sku: null,
    barcode: null,
    unit_name: null,
    vat_flag: 'V',
    price: null,
    category_id: 'sub',
  },
]

const NODES = [
  { id: 'root', name_th: 'เครื่องเขียน_สำนักงาน', parent_id: null },
  { id: 'sub', name_th: 'กาว/เทป', parent_id: 'root' },
]

/** sku ที่สตอกเดิมใช้ไปแล้ว — ตัวออกรหัสต้องเดินเลขต่อจากนี่ */
const EXISTING_SKUS = [{ sku: 'ST133' }, { sku: 'HH695' }]

beforeEach(() => {
  calls.length = 0
  queues = {
    'products.select': [
      { data: PRODUCTS, error: null },      // ครั้งที่ 1: รายการที่จะส่งออก
      { data: EXISTING_SKUS, error: null }, // ครั้งที่ 2: sku ที่ใช้ไปแล้ว
    ],
    'products.update': [{ data: null, error: null }],
    'taxonomy_nodes.select': [{ data: NODES, error: null }],
  }
})

describe('GET /api/pos-export', () => {
  it('คัดด้วย source + dedup_bucket + ยังไม่เคยส่งออก ไม่ใช่ status', async () => {
    const { GET } = await import('@/app/api/pos-export/route')
    await GET(new NextRequest('http://127.0.0.1:3000/api/pos-export'))

    const productCall = calls.find((c) => c.table === 'products')
    expect(productCall).toBeDefined()

    const eqFilters = productCall!.filters.filter((f) => f.kind === 'eq')
    expect(eqFilters).toEqual(
      expect.arrayContaining([
        { kind: 'eq', column: "metadata->>source", value: 'import_wizard' },
        { kind: 'eq', column: "metadata->>dedup_bucket", value: 'new' },
      ])
    )
    expect(productCall!.filters).toContainEqual({
      kind: 'is', column: 'pos_exported_at', value: null,
    })
    // status พลิกเป็น approved ตอน verify จึงใช้กรองไม่ได้
    expect(eqFilters.some((f) => f.column === 'status')).toBe(false)
  })

  it('คืนยอดรวมและจำนวนช่องบังคับที่ยังขาด', async () => {
    const { GET } = await import('@/app/api/pos-export/route')
    const response = await GET(new NextRequest('http://127.0.0.1:3000/api/pos-export'))
    const body = await response.json()

    expect(body.success).toBe(true)
    expect(body.total).toBe(2)
    // รหัสถูกออกให้แล้วตอน preview หน่วยตกเป็น ชิ้น เหลือแค่ราคาที่ยังไม่มี
    expect(body.issues).toEqual({
      missingItemId: 0, missingBarcode: 0, missingItemName: 0,
      missingUnitName: 0, missingItemGroup: 0, missingPrice: 1,
    })
  })

  it('หน้าตัวอย่างแสดงรหัสที่จะถูกออกให้ ไม่ใช่ช่องว่าง', async () => {
    const { GET } = await import('@/app/api/pos-export/route')
    const response = await GET(new NextRequest('http://127.0.0.1:3000/api/pos-export'))
    const body = await response.json()

    const ids = body.preview.map((r: { itemId: string }) => r.itemId)
    expect(ids).toContain('HH001')  // ของเดิมไม่ถูกแตะ
    expect(ids).toContain('ST134')  // ออกใหม่ เดินต่อจาก ST133
  })

  it('ตัวอย่างไม่เขียนอะไรลงฐานข้อมูล', async () => {
    const { GET } = await import('@/app/api/pos-export/route')
    await GET(new NextRequest('http://127.0.0.1:3000/api/pos-export'))

    expect(calls.some((c) => c.op === 'update')).toBe(false)
  })

  it('หน่วยที่ว่างตกเป็น ชิ้น', async () => {
    const { GET } = await import('@/app/api/pos-export/route')
    const response = await GET(new NextRequest('http://127.0.0.1:3000/api/pos-export'))
    const body = await response.json()

    expect(body.preview.map((r: { unitName: string }) => r.unitName)).toEqual(
      expect.arrayContaining(['ใบ', 'ชิ้น'])
    )
  })

  it('ตัวอย่างที่คืนมาแทน _ ในชื่อหมวดด้วยช่องว่างแล้ว', async () => {
    const { GET } = await import('@/app/api/pos-export/route')
    const response = await GET(new NextRequest('http://127.0.0.1:3000/api/pos-export'))
    const body = await response.json()

    expect(body.preview[0].itemGroup).toBe('เครื่องเขียน สำนักงาน')
    expect(body.preview[0].subGroup1).toBe('กาว/เทป')
  })
})

describe('POST /api/pos-export', () => {
  const post = async (body: unknown) => {
    const { POST } = await import('@/app/api/pos-export/route')
    return POST(
      new NextRequest('http://127.0.0.1:3000/api/pos-export', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body),
      })
    )
  }

  it('คืนไฟล์เป็นข้อความ ไม่ใช่ JSON', async () => {
    const response = await post({})
    expect(response.headers.get('content-type')).toContain('text/plain')
    expect(response.headers.get('content-type')).toContain('charset=utf-8')
    expect(response.headers.get('content-disposition')).toContain('attachment')
  })

  it('เนื้อไฟล์มีหัวตารางและครบทุกแถว', async () => {
    const response = await post({})
    const text = await response.text()
    const lines = text.split('\r\n').filter(Boolean)

    expect(lines).toHaveLength(3) // หัวตาราง + 2 สินค้า
    expect(lines[0].split('\t')).toHaveLength(9)
    expect(text.endsWith('\r\n')).toBe(true)
    expect(text.charCodeAt(0)).not.toBe(0xfeff)
  })

  it('เขียนรหัสที่ออกใหม่ลง products.sku', async () => {
    await post({})

    const skuUpdate = calls.find(
      (c) => c.op === 'update' && (c.payload as Record<string, unknown>)?.sku !== undefined
    )
    expect(skuUpdate).toBeDefined()
    expect(skuUpdate!.payload).toEqual({ sku: 'ST134' })
    expect(skuUpdate!.filters).toContainEqual({ kind: 'eq', column: 'id', value: 'p2' })
  })

  it('ไม่เขียนทับรหัสของรายการที่มีอยู่แล้ว', async () => {
    await post({})

    const skuUpdates = calls.filter(
      (c) => c.op === 'update' && (c.payload as Record<string, unknown>)?.sku !== undefined
    )
    expect(skuUpdates).toHaveLength(1) // p1 มี HH001 อยู่แล้ว ไม่ถูกแตะ
  })

  it('รหัสในไฟล์ตรงกับรหัสที่เขียนลงฐานข้อมูล', async () => {
    const response = await post({})
    const text = await response.text()

    expect(text).toContain('ST134')
  })

  it('ทำเครื่องหมาย pos_exported_at ให้แถวที่ส่งออกไป', async () => {
    await post({})

    const marked = calls.find(
      (c) => c.op === 'update' && (c.payload as Record<string, unknown>)?.pos_exported_at !== undefined
    )
    expect(marked).toBeDefined()
    expect(marked!.filters).toContainEqual({
      kind: 'in', column: 'id', value: ['p1', 'p2'],
    })
  })

  it('เขียนรหัสก่อนแล้วค่อยทำเครื่องหมายส่งออก', async () => {
    await post({})

    const updates = calls.filter((c) => c.op === 'update')
    const skuAt = updates.findIndex((c) => (c.payload as Record<string, unknown>)?.sku !== undefined)
    const markAt = updates.findIndex(
      (c) => (c.payload as Record<string, unknown>)?.pos_exported_at !== undefined
    )
    // ถ้าเขียนรหัสพลาด ต้องยังไม่มีอะไรถูกตีว่าส่งออกแล้ว ของจะได้ค้างในคิว
    expect(skuAt).toBeLessThan(markAt)
  })

  it('mark_exported=false โหลดไฟล์ได้โดยไม่แตะฐานข้อมูล', async () => {
    await post({ mark_exported: false })

    expect(calls.some((c) => c.op === 'update')).toBe(false)
  })

  it('ไม่มีอะไรให้ส่งออกก็ตอบ 409 พร้อมข้อความไทย ไม่ใช่ไฟล์เปล่า', async () => {
    queues['products.select'] = [
      { data: [], error: null },
      { data: [], error: null },
    ]

    const response = await post({})
    expect(response.status).toBe(409)

    const body = await response.json()
    expect(body.success).toBe(false)
    expect(body.error).toContain('ไม่มีสินค้า')
  })
})

describe('middleware', () => {
  it('กัน GET /api/pos-export ไว้ เพราะอ่านด้วย service role', () => {
    const source = readFileSync(
      path.join(__dirname, '..', '..', 'middleware.ts'), 'utf8'
    )
    const gated = source.match(/GATED_READ_PATHS = new Set\(\[([^\]]*)\]/)
    expect(gated).not.toBeNull()
    expect(gated![1]).toContain('/api/pos-export')
  })
})
