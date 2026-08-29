/**
 * ตัวเลข "งานค้าง" บนหน้าแรกเคยนับจาก status 'pending' ซึ่งไม่มีอยู่จริงในฐานข้อมูล
 * pipeline ที่เขียนข้อมูลจริง (api/import/commit) ใช้ pending_review_dedup /
 * pending_review_category เท่านั้น เทสต์ชุดนี้ล็อกไว้ว่าการนับต้องอ้างอิงสองค่านั้น
 */

type Filter = { op: 'eq' | 'in' | 'gte'; column: string; value: unknown }
type Query = { table: string; filters: Filter[] }

const queries: Query[] = []

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
        range: () => chain,
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
          Promise.resolve({ data: [], count: countFor(query), error: null }).then(resolve)
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
