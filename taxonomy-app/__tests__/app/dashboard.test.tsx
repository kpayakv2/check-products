/**
 * หน้าแรกเคยโชว์ตัวเลขที่ฝังไว้ในโค้ด — Accuracy 99.8%, Latency 12ms และป้าย
 * "+8.2%" ใต้การ์ดสถิติ — ทั้งที่ไม่มีอะไรวัดค่าพวกนี้เลย เทสต์ชุดนี้กันไม่ให้
 * ตัวเลขแบบนั้นกลับมา และล็อกว่าเลขที่แสดงต้องมาจาก getDashboardStats
 */
import { render, screen } from '@testing-library/react'

const stats = {
  totalCategories: 134,
  totalSynonyms: 28,
  pendingDedup: 147,
  pendingCategory: 221,
  pendingProducts: 368,
  approvedProducts: 3103,
  rejectedProducts: 37,
  duplicatePairs: 1781,
  duplicatePairsReviewed: 1381,
  reviewsToday: 4,
  recheckAgreement: { total: 3103, agreed: 2469 }
}

jest.mock('@/components/Layout/Sidebar', () => ({ __esModule: true, default: () => null }))
jest.mock('@/components/Layout/Header', () => ({ __esModule: true, default: () => null }))

jest.mock('@/utils/supabase', () => ({
  DatabaseService: {
    getDashboardStats: jest.fn(async () => stats)
  }
}))

import Dashboard from '@/app/page'

describe('หน้าแรก', () => {
  it('แสดงงานค้างจริงแยกตามด่านที่ต้องไปตรวจ', async () => {
    render(<Dashboard />)

    expect(await screen.findByTestId('pending-total')).toHaveTextContent('368')
    expect(screen.getByTestId('pending-dedup')).toHaveTextContent('147')
    expect(screen.getByTestId('pending-category')).toHaveTextContent('221')
  })

  it('แสดงสัดส่วนที่ AI ตรงกับคนตามข้อมูลจริง', async () => {
    render(<Dashboard />)

    // 2,469 จาก 3,103 = 79.6%
    expect(await screen.findByTestId('recheck-agreement')).toHaveTextContent('79.6%')
  })

  it('ไม่มีตัวเลขที่ฝังไว้ในโค้ดหลงเหลือบนหน้า', async () => {
    const { container } = render(<Dashboard />)
    await screen.findByTestId('pending-total')

    const text = container.textContent || ''
    expect(text).not.toContain('99.8')
    expect(text).not.toContain('12ms')
    expect(text).not.toMatch(/\+\d+(\.\d+)?%/)
  })

  it('ไม่มีลิงก์ไปหน้ารายงานที่ถูกยุบเข้ามาแล้ว', async () => {
    const { container } = render(<Dashboard />)
    await screen.findByTestId('pending-total')

    expect(container.querySelector('[href="/reports"]')).toBeNull()
  })
})
