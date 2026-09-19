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
 * คิวผลลัพธ์ต่อ `table.op` เพราะ route ยิง products หลายครั้ง:
 * ครั้งแรกเอารายการที่จะส่งออก ที่เหลือเอา sku ทั้งหมดไปคำนวณรหัสใหม่
 *
 * ทุกการอ่านไล่ทีละหน้าด้วย .range() — หนึ่งหน้าคือหนึ่ง select หนึ่งคิว
 * หน้าที่ได้แถวไม่ครบ 1,000 คือหน้าสุดท้าย
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
    not: self,
    order: (column: string) => {
      call.filters.push({ kind: 'order', column })
      return chain
    },
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
    range: (from: number, to: number) => {
      call.filters.push({ kind: 'range', value: [from, to] })
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

/**
 * PostgREST ตัดผลลัพธ์ที่ไม่ใส่ range ทิ้งที่ max_rows (1000 ในรีโปนี้) แล้วตอบ 206
 * โดยที่ supabase-js ไม่ยก error ขึ้นมาเลย — ต้องไล่หน้าเอง ไม่งั้นข้อมูลขาดแบบเงียบ ๆ
 */
describe('ไล่ดึงทีละหน้า (max_rows = 1000)', () => {
  const PAGE_SIZE = 1000

  /** สินค้าปลอมจำนวนมาก สำหรับทดสอบว่าหน้าเต็มแล้วยิงหน้าถัดไปจริง */
  const makeProducts = (count: number, withSku: boolean) =>
    Array.from({ length: count }, (_, index) => ({
      id: `bulk-${String(index).padStart(4, '0')}`,
      name_th: `สินค้าชุดใหญ่ ${String(index).padStart(4, '0')}`,
      sku: withSku ? `HH${index + 1}` : null,
      barcode: null,
      unit_name: 'ชิ้น',
      vat_flag: 'V',
      price: 10,
      category_id: 'sub',
    }))

  const selectsOn = (table: string) =>
    calls.filter((c) => c.table === table && c.op === 'select')

  const rangeOf = (call: Call) =>
    call.filters.find((f) => f.kind === 'range')?.value

  it('ต่อทุกหน้าของ sku ที่ใช้ไปแล้วเข้าด้วยกัน ไม่ใช่หยุดที่หน้าแรก', async () => {
    // เลขสูงสุดของ prefix ST อยู่หน้าที่สอง — ถ้าอ่านแค่หน้าแรกจะออกรหัสชนของเดิม
    queues['products.select'] = [
      { data: PRODUCTS, error: null },
      { data: Array.from({ length: PAGE_SIZE }, (_, i) => ({ sku: `ZZ${i + 1}` })), error: null },
      { data: [{ sku: 'ST900' }], error: null },
    ]

    const { GET } = await import('@/app/api/pos-export/route')
    const response = await GET(new NextRequest('http://127.0.0.1:3000/api/pos-export'))
    const body = await response.json()

    const selects = selectsOn('products')
    expect(selects).toHaveLength(3) // 1 หน้าของรายการ + 2 หน้าของ sku
    expect(rangeOf(selects[1])).toEqual([0, PAGE_SIZE - 1])
    expect(rangeOf(selects[2])).toEqual([PAGE_SIZE, PAGE_SIZE * 2 - 1])

    expect(body.preview.map((r: { itemId: string }) => r.itemId)).toContain('ST901')
  })

  it('ต่อทุกหน้าของรายการที่รอส่งออก ไม่ใช่รายงานแค่ 1,000 แถวแรก', async () => {
    queues['products.select'] = [
      { data: makeProducts(PAGE_SIZE, true), error: null },
      { data: [PRODUCTS[1]], error: null }, // หน้าสุดท้ายไม่เต็ม
      { data: EXISTING_SKUS, error: null },
    ]

    const { GET } = await import('@/app/api/pos-export/route')
    const response = await GET(new NextRequest('http://127.0.0.1:3000/api/pos-export'))
    const body = await response.json()

    expect(body.total).toBe(PAGE_SIZE + 1)

    const selects = selectsOn('products')
    expect(rangeOf(selects[0])).toEqual([0, PAGE_SIZE - 1])
    expect(rangeOf(selects[1])).toEqual([PAGE_SIZE, PAGE_SIZE * 2 - 1])
  })

  it('ทุก query ที่ไล่หน้ามี .order() กำกับ ไม่งั้นแถวข้ามหรือซ้ำระหว่างหน้า', async () => {
    const { GET } = await import('@/app/api/pos-export/route')
    await GET(new NextRequest('http://127.0.0.1:3000/api/pos-export'))

    const paged = calls.filter(
      (c) => c.op === 'select' && c.filters.some((f) => f.kind === 'range')
    )
    expect(paged.length).toBeGreaterThan(0)
    for (const call of paged) {
      expect(call.filters.some((f) => f.kind === 'order')).toBe(true)
    }
  })

  it('หน้าไหน error ก็โยนออกไป ไม่กลืนแล้วคำนวณต่อด้วยข้อมูลครึ่ง ๆ', async () => {
    queues['products.select'] = [
      { data: null, error: { message: 'boom' } },
    ]

    const { GET } = await import('@/app/api/pos-export/route')
    const response = await GET(new NextRequest('http://127.0.0.1:3000/api/pos-export'))

    expect(response.status).toBeGreaterThanOrEqual(400)
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

  /**
   * .in() ยัด uuid ทุกตัวลง query string ของ URL — ของจริง 221 รายการก็ทะลุ
   * บัฟเฟอร์ ~8KB ของ gateway แล้วได้ 414 กลับมา รหัสถูกเขียนแต่ไม่มีอะไรถูกตี
   * ว่าส่งออก คิวจึงค้างวนชนรหัสซ้ำทุกรอบ
   */
  it('ตีว่าส่งออกแล้วทีละก้อน ไม่ยัด id ทั้งหมดลง .in() ครั้งเดียว', async () => {
    const MANY = 60
    const BATCH = 25
    const bulk = Array.from({ length: MANY }, (_, index) => ({
      id: `bulk-${String(index).padStart(4, '0')}`,
      name_th: `สินค้าชุดใหญ่ ${String(index).padStart(4, '0')}`,
      sku: null,
      barcode: null,
      unit_name: 'ชิ้น',
      vat_flag: 'V',
      price: 10,
      category_id: 'sub',
    }))
    queues['products.select'] = [
      { data: bulk, error: null },
      { data: EXISTING_SKUS, error: null },
    ]

    await post({})

    const updates = calls.filter((c) => c.op === 'update')
    const marks = updates.filter(
      (c) => (c.payload as Record<string, unknown>)?.pos_exported_at !== undefined
    )

    expect(marks).toHaveLength(Math.ceil(MANY / BATCH)) // 60 → 3 ก้อน
    for (const mark of marks) {
      const ids = mark.filters.find((f) => f.kind === 'in')?.value as string[]
      expect(ids.length).toBeLessThanOrEqual(BATCH)
    }

    // ครบทุกตัว ไม่มีแถวไหนตกหล่นจากการแบ่งก้อน
    const marked = marks.flatMap(
      (mark) => mark.filters.find((f) => f.kind === 'in')?.value as string[]
    )
    expect(marked).toEqual(bulk.map((product) => product.id))

    // รหัสต้องเขียนครบทุกก้อนก่อน ค่อยเริ่มตีว่าส่งออก
    const lastSkuWrite = updates.map((c) =>
      (c.payload as Record<string, unknown>)?.sku !== undefined
    ).lastIndexOf(true)
    const firstMark = updates.findIndex(
      (c) => (c.payload as Record<string, unknown>)?.pos_exported_at !== undefined
    )
    expect(lastSkuWrite).toBeLessThan(firstMark)
    expect(updates.filter((c) => (c.payload as Record<string, unknown>)?.sku !== undefined))
      .toHaveLength(MANY)
  })

  it('ก้อนไหนพัง โยน error ออกไป ไม่ตอบว่าสำเร็จ', async () => {
    queues['products.update'] = [
      { data: null, error: null },
      { data: null, error: { message: 'update failed' } },
    ]

    const response = await post({})
    expect(response.status).toBeGreaterThanOrEqual(400)
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
