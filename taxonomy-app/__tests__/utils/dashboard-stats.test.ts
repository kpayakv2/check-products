/**
 * ตัวเลข "งานค้าง" บนหน้าแรกเคยนับจาก status 'pending' ซึ่งไม่มีอยู่จริงในฐานข้อมูล
 * pipeline ที่เขียนข้อมูลจริง (api/import/commit) ใช้ pending_review_dedup /
 * pending_review_category เท่านั้น เทสต์ชุดนี้ล็อกไว้ว่าการนับต้องอ้างอิงสองค่านั้น
 */

type Filter = { op: 'eq' | 'in' | 'gte' | 'or'; column: string; value: unknown }
type Query = { table: string; filters: Filter[]; range?: [number, number] }

const queries: Query[] = []

/** แถวตัวอย่างที่ query ไหนก็ตามจะได้กลับไป */
const ROWS = [{ id: 'p-1', name_th: 'ปากกาลูกลื่น', status: 'approved' }]

/** จำนวนแถวปลอมที่ตอบกลับตามชุดเงื่อนไขที่ query นั้นใช้ */
const countFor = (query: Query): number => {
  if (query.table !== 'products') return 0
  const status = query.filters.find(f => f.column === 'status')
  if (!status) return 0
  if (status.op === 'in') {
    const values = status.value as string[]
    return values.reduce((sum, v) => sum + (STATUS_COUNTS[v] ?? 0), 0)
  }
  return STATUS_COUNTS[status.value as string] ?? 0
}

const STATUS_COUNTS: Record<string, number> = {
  pending_review_dedup: 147,
  pending_review_category: 221,
  approved: 3103,
  rejected: 37,
  pending: 0
}

jest.mock('@supabase/supabase-js', () => ({
  createClient: () => ({
    from: (table: string) => {
      const query: Query = { table, filters: [] }
      queries.push(query)
      const chain: any = {
        select: () => chain,
        order: () => chain,
        limit: () => chain,
        range: (from: number, to: number) => {
          query.range = [from, to]
          return chain
        },
        or: (expression: string) => {
          query.filters.push({ op: 'or', column: 'or', value: expression })
          return chain
        },
        eq: (column: string, value: unknown) => {
          query.filters.push({ op: 'eq', column, value })
          return chain
        },
        in: (column: string, value: unknown) => {
          query.filters.push({ op: 'in', column, value })
          return chain
        },
        gte: (column: string, value: unknown) => {
          query.filters.push({ op: 'gte', column, value })
          return chain
        },
        then: (resolve: (v: unknown) => unknown) =>
          Promise.resolve({ data: ROWS, count: countFor(query), error: null }).then(resolve)
      }
      return chain
    }
  })
}))

import { DatabaseService } from '@/utils/supabase'

const productQueries = () => queries.filter(q => q.table === 'products')

beforeEach(() => {
  queries.length = 0
})

describe('getDashboardStats', () => {
  it('นับงานค้างจากทั้งสองด่านที่ pipeline เขียนจริง ไม่ใช่ status pending', async () => {
    const stats = await DatabaseService.getDashboardStats()

    expect(stats.pendingProducts).toBe(368)
  })

  it('ไม่มี query ไหนกรองด้วย status pending อีกแล้ว', async () => {
    await DatabaseService.getDashboardStats()

    const usesPending = productQueries().some(q =>
      q.filters.some(f => f.column === 'status' && f.value === 'pending')
    )
    expect(usesPending).toBe(false)
  })

  it('ยังนับสินค้าที่อนุมัติแล้วแยกจากงานค้าง', async () => {
    const stats = await DatabaseService.getDashboardStats()

    expect(stats.approvedProducts).toBe(3103)
  })
})

describe('getProducts', () => {
  it('รับ status หลายค่าแล้วกรองด้วย in ไม่ใช่ eq', async () => {
    await DatabaseService.getProducts(['pending_review_dedup', 'pending_review_category'])

    const statusFilters = productQueries()[0].filters.filter(f => f.column === 'status')
    expect(statusFilters).toEqual([
      { op: 'in', column: 'status', value: ['pending_review_dedup', 'pending_review_category'] }
    ])
  })

  it('รับ status ค่าเดียวแล้วยังกรองด้วย eq เหมือนเดิม', async () => {
    await DatabaseService.getProducts('approved')

    const statusFilters = productQueries()[0].filters.filter(f => f.column === 'status')
    expect(statusFilters).toEqual([{ op: 'eq', column: 'status', value: 'approved' }])
  })
})

describe('searchProducts', () => {
  it('ค้นหาที่ฝั่งฐานข้อมูล ไม่ใช่ดึงมากรองในเบราว์เซอร์', async () => {
    await DatabaseService.searchProducts({ search: 'ปากกา' })

    const orFilter = productQueries()[0].filters.find(f => f.op === 'or')
    expect(orFilter).toBeDefined()
    expect(String(orFilter!.value)).toContain('ปากกา')
  })

  it('ตัดอักขระที่ทำให้ตัวกรองของ PostgREST เพี้ยนออกจากคำค้น', async () => {
    await DatabaseService.searchProducts({ search: 'ปากกา,ลูกลื่น(แดง)' })

    // จุลภาคใน expression เป็นตัวคั่นเงื่อนไขของ PostgREST เอง ที่ต้องไม่มีคือใน "คำค้น"
    const orFilter = productQueries()[0].filters.find(f => f.op === 'or')
    const terms = String(orFilter!.value).match(/%[^%]*%/g) ?? []
    expect(terms).not.toHaveLength(0)
    terms.forEach(term => expect(term).not.toMatch(/[,()]/))
  })

  it('ขอข้อมูลทีละหน้าตาม offset ที่ส่งมา', async () => {
    await DatabaseService.searchProducts({ limit: 25, offset: 50 })

    expect(productQueries()[0].range).toEqual([50, 74])
  })

  it('คืนจำนวนทั้งหมดมาด้วย เพื่อให้หน้าเว็บรู้ว่ามีกี่หน้า', async () => {
    const result = await DatabaseService.searchProducts({ status: 'approved' })

    expect(result.total).toBe(3103)
    expect(result.products).toHaveLength(1)
  })
})

describe('getProductStatusCounts', () => {
  it('นับแยกรายสถานะโดยไม่ต้องดึงแถวจริงมาทั้งหมด', async () => {
    const counts = await DatabaseService.getProductStatusCounts()

    expect(counts.approved).toBe(3103)
    expect(counts.pending_review_dedup).toBe(147)
    expect(counts.pending_review_category).toBe(221)
    expect(counts.rejected).toBe(37)
  })
})
