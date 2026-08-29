/**
 * หน้า /products เคยตั้งตัวกรองเริ่มต้นเป็น status 'pending' ซึ่งไม่มีอยู่จริงในฐานข้อมูล
 * ผลคือเปิดหน้ามาว่างเปล่าถาวร และตัวนับ "Pending" ขึ้น 0 ทั้งที่มีงานค้างจริง
 */
import { render, screen, waitFor } from '@testing-library/react'
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
  },
  {
    id: 'p-3',
    name_th: 'แชมพู',
    status: 'pending_review_category',
    created_at: '2026-08-03T00:00:00Z',
    updated_at: '2026-08-03T00:00:00Z'
  }
]

jest.mock('@/components/Layout/Sidebar', () => ({ __esModule: true, default: () => null }))
jest.mock('@/components/Layout/Header', () => ({ __esModule: true, default: () => null }))
jest.mock('react-hot-toast', () => ({ toast: { error: jest.fn(), success: jest.fn() } }))

jest.mock('@/utils/supabase', () => {
  const actual = jest.requireActual('@/utils/product-status')
  return {
    ...actual,
    DatabaseService: {
      getProducts: jest.fn(async () => products),
      getTaxonomyTree: jest.fn(async () => [])
    }
  }
})

import ProductsPage from '@/app/products/page'

describe('หน้ารายการสินค้า', () => {
  it('เปิดมาแล้วเห็นสินค้าในสตอก ไม่ใช่หน้าว่าง', async () => {
    render(<ProductsPage />)

    expect(await screen.findByText('ปากกาลูกลื่น')).toBeInTheDocument()
  })

  it('ตัวนับงานค้างรวมทั้งสองด่านที่รอคนตรวจ', async () => {
    render(<ProductsPage />)

    const counter = await screen.findByTestId('pending-count')
    await waitFor(() => expect(counter).toHaveTextContent('2'))
  })
})
