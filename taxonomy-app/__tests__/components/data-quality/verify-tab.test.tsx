/**
 * แท็บ Verify คือที่ทำงานของงานตรวจ 368 รายการ แต่เดิมเขียน DB ตรงด้วย anon key
 * ซึ่ง RLS กรองแถวทิ้งเงียบ ๆ — กดแล้วไม่มีอะไรเกิดขึ้น และไม่มี error ให้เห็นด้วย
 * เทสต์ชุดนี้ล็อกว่าทุกคำตัดสินต้องผ่าน API route ที่ใช้ service role และต้องเชื่อ
 * ผลลัพธ์จาก route จริง ๆ ไม่ใช่ตัดรายการออกจากจอไปก่อน
 */
import { render, screen, fireEvent, waitFor } from '@testing-library/react'

const products = [
  {
    id: '11111111-1111-4111-8111-111111111111',
    name_th: 'ปากกาลูกลื่น สีน้ำเงิน',
    status: 'pending_review_dedup',
    confidence_score: 0.8,
    metadata: { duplicate_of: 'ปากกาลูกลื่น น้ำเงิน', similarity_score: 0.93, clean_name: 'ปากกาลูกลื่น' },
    created_at: '2026-08-01T00:00:00Z'
  }
]

const taxonomy = [{ id: '22222222-2222-4222-8222-222222222222', name_th: 'เครื่องเขียน' }]

jest.mock('@/utils/supabase', () => {
  const makeChain = (table: string, options?: { head?: boolean }) => {
    const chain: any = {
      eq: () => chain,
      order: () => chain,
      limit: () => chain,
      then: (resolve: (value: unknown) => unknown) => {
        if (options?.head) return Promise.resolve({ count: 1 }).then(resolve)
        if (table === 'taxonomy_nodes') return Promise.resolve({ data: taxonomy }).then(resolve)
        return Promise.resolve({ data: products }).then(resolve)
      }
    }
    return chain
  }
  return {
    supabase: {
      from: (table: string) => ({
        select: (_columns?: string, options?: { head?: boolean }) => makeChain(table, options)
      })
    }
  }
})

jest.mock('react-hot-toast', () => ({ toast: { error: jest.fn(), success: jest.fn() } }))

import { toast } from 'react-hot-toast'
import VerifyTab from '@/components/data-quality/VerifyTab'

const toastError = toast.error as jest.Mock
let fetchMock: jest.Mock

const mockFetch = (response: { ok: boolean; status?: number; body?: unknown }) => {
  fetchMock = jest.fn(async () => ({
    ok: response.ok,
    status: response.status ?? (response.ok ? 200 : 500),
    json: async () => response.body ?? { success: response.ok }
  }))
  global.fetch = fetchMock as unknown as typeof fetch
}

const verifyCalls = () => fetchMock.mock.calls.filter(call => String(call[0]).includes('/api/verify'))

const bodyOf = (index: number) => JSON.parse(String((verifyCalls()[index][1] as RequestInit).body))

beforeEach(() => {
  toastError.mockClear()
  mockFetch({ ok: true, body: { success: true } })
})

describe('ด่าน 1 ตรวจของซ้ำ', () => {
  it('ส่งคำตัดสิน "ไม่ซ้ำ" ผ่าน API route ที่ใช้ service role', async () => {
    render(<VerifyTab />)

    fireEvent.click(await screen.findByTestId('keep-btn'))

    await waitFor(() => expect(verifyCalls().length).toBe(1))
    expect(bodyOf(0)).toMatchObject({ product_id: products[0].id, action: 'keep' })
  })

  it('ส่งคำตัดสิน "ซ้ำจริง" ให้ route ลบให้ ไม่ลบเองด้วย anon key', async () => {
    render(<VerifyTab />)

    fireEvent.click(await screen.findByTestId('discard-btn'))

    await waitFor(() => expect(verifyCalls().length).toBe(1))
    expect(bodyOf(0)).toMatchObject({ product_id: products[0].id, action: 'discard' })
  })

  it('คงรายการไว้และบอกผู้ใช้ เมื่อ route บอกว่าไม่มีแถวไหนเปลี่ยน', async () => {
    mockFetch({ ok: false, status: 404, body: { success: false, error: 'ไม่พบสินค้ารายการนี้แล้ว' } })
    render(<VerifyTab />)

    fireEvent.click(await screen.findByTestId('keep-btn'))

    await waitFor(() => expect(toastError).toHaveBeenCalled())
    expect(screen.getByText('ปากกาลูกลื่น สีน้ำเงิน')).toBeInTheDocument()
  })
})

describe('ด่าน 2 ยืนยันหมวดหมู่', () => {
  const openCategoryTab = async () => {
    fireEvent.click(await screen.findByTestId('tab-verify-category'))
  }

  it('ส่งหมวดที่คนเลือกไปกับคำขอ', async () => {
    render(<VerifyTab />)
    await openCategoryTab()

    fireEvent.change(await screen.findByTestId('category-select'), {
      target: { value: taxonomy[0].id }
    })

    await waitFor(() => expect(verifyCalls().length).toBe(1))
    expect(bodyOf(0)).toMatchObject({ action: 'confirm_category', category_id: taxonomy[0].id })
  })

  it('ไม่ยิงคำขอเมื่อยังไม่ได้เลือกหมวดหมู่ (เดิมส่งค่าว่างไปเงียบ ๆ)', async () => {
    render(<VerifyTab />)
    await openCategoryTab()

    fireEvent.click(await screen.findByTestId('confirm-category-btn'))

    await waitFor(() => expect(toastError).toHaveBeenCalled())
    expect(verifyCalls()).toHaveLength(0)
  })
})
