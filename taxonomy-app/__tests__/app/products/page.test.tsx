/**
 * /products เป็น "หน้าดูสตอก" อย่างเดียว งานตรวจอยู่ที่ /data-quality ที่เดียว
 *
 * สองเรื่องที่เทสต์ชุดนี้ล็อกไว้:
 * - ไม่มีปุ่มอนุมัติ/ปฏิเสธหลงเหลือในหน้านี้อีก
 * - การค้นและแบ่งหน้าเกิดที่ฐานข้อมูล ไม่ใช่ดึง 50 แถวมากรองในเบราว์เซอร์
 *   (สตอกจริงมี 3,103 แถว การกรองในหน่วยความจำจึงค้นไม่เจอของส่วนใหญ่)
 */
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import type { Product } from '@/utils/supabase'

const products: Product[] = [
  {
    id: 'p-1',
    name_th: 'ปากกาลูกลื่น',
    status: 'approved',
    created_at: '2026-08-01T00:00:00Z',
    updated_at: '2026-08-01T00:00:00Z'
  },
  {
    id: 'p-2',
    name_th: 'ยาสีฟัน',
    status: 'pending_review_dedup',
    created_at: '2026-08-02T00:00:00Z',
    updated_at: '2026-08-02T00:00:00Z'
  }
]

const searchProducts = jest.fn(async () => ({ products, total: 3103 }))
const getProductStatusCounts = jest.fn(async () => ({
  pending_review_dedup: 147,
  pending_review_category: 221,
  approved: 3103,
  rejected: 37,
  draft: 0
}))

jest.mock('@/components/Layout/Sidebar', () => ({ __esModule: true, default: () => null }))
jest.mock('@/components/Layout/Header', () => ({ __esModule: true, default: () => null }))
jest.mock('react-hot-toast', () => ({ toast: { error: jest.fn(), success: jest.fn() } }))

jest.mock('@/utils/supabase', () => {
  const actual = jest.requireActual('@/utils/product-status')
  return {
    ...actual,
    DatabaseService: {
      searchProducts: (...args: unknown[]) => searchProducts(...(args as [])),
      getProductStatusCounts: () => getProductStatusCounts(),
      getTaxonomyTree: jest.fn(async () => [])
    }
  }
})

import ProductsPage from '@/app/products/page'

beforeEach(() => {
  searchProducts.mockClear()
  getProductStatusCounts.mockClear()
})

describe('หน้าดูสตอก /products', () => {
  it('เปิดมาแล้วเห็นสินค้าในสตอก ไม่ใช่หน้าว่าง', async () => {
    render(<ProductsPage />)

    expect(await screen.findByText('ปากกาลูกลื่น')).toBeInTheDocument()
  })

  it('ตัวนับงานค้างนับจากทั้งฐานข้อมูล ไม่ใช่เฉพาะหน้าที่โหลดมา', async () => {
    render(<ProductsPage />)

    const counter = await screen.findByTestId('pending-count')
    await waitFor(() => expect(counter).toHaveTextContent('368'))
  })

  it('ไม่มีปุ่มอนุมัติหรือปฏิเสธในหน้านี้แล้ว', async () => {
    render(<ProductsPage />)
    fireEvent.click(await screen.findByText('ยาสีฟัน'))

    await screen.findByTestId('product-detail')
    expect(screen.queryByRole('button', { name: /อนุมัติ|approval|approve/i })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /ปฏิเสธ|reject/i })).not.toBeInTheDocument()
  })

  it('พิมพ์คำค้นแล้วไปค้นที่ฐานข้อมูล', async () => {
    render(<ProductsPage />)
    await screen.findByText('ปากกาลูกลื่น')

    fireEvent.change(screen.getByTestId('product-search'), { target: { value: 'ยาสีฟัน' } })

    await waitFor(() =>
      expect(searchProducts).toHaveBeenCalledWith(expect.objectContaining({ search: 'ยาสีฟัน' }))
    )
  })

  it('กดหน้าถัดไปแล้วขอข้อมูลชุดใหม่ด้วย offset', async () => {
    render(<ProductsPage />)
    await screen.findByText('ปากกาลูกลื่น')

    fireEvent.click(screen.getByTestId('next-page'))

    await waitFor(() =>
      expect(searchProducts).toHaveBeenCalledWith(expect.objectContaining({ offset: 50 }))
    )
  })
})
