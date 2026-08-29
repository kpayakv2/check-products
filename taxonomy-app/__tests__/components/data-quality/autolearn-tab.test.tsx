/**
 * ปุ่มลบกฎในแท็บ Auto-learn เคยลบด้วย anon key ที่ RLS ปิดอยู่ — ได้ 204 กลับมา
 * โดยไม่มีแถวไหนถูกลบ แล้วโค้ดก็ตัดการ์ดออกจาก state ทันที ดูเหมือนลบสำเร็จ
 * จนกว่าจะกดรีเฟรชแล้วเจอกฎเดิมกลับมา
 */
import { render, screen, fireEvent, waitFor } from '@testing-library/react'

const rules = [
  {
    id: '33333333-3333-4333-8333-333333333333',
    code: 'KW-001',
    name: 'ปากกา',
    keywords: ['ปากกา', 'ลูกลื่น'],
    category_id: '22222222-2222-4222-8222-222222222222',
    match_type: 'auto_learned',
    created_at: '2026-08-01T00:00:00Z',
    taxonomy_nodes: { name_th: 'เครื่องเขียน' }
  }
]

jest.mock('@/utils/supabase', () => {
  const makeChain = () => {
    const chain: any = {
      eq: () => chain,
      order: () => Promise.resolve({ data: rulesRef.current, error: null })
    }
    return chain
  }
  const rulesRef = { current: [] as unknown[] }
  return {
    __rulesRef: rulesRef,
    supabase: { from: () => ({ select: () => makeChain() }) }
  }
})

jest.mock('react-hot-toast', () => ({ toast: { error: jest.fn(), success: jest.fn() } }))

import { toast } from 'react-hot-toast'
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { __rulesRef } = require('@/utils/supabase')
import AutoLearnTab from '@/components/data-quality/AutoLearnTab'

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

beforeEach(() => {
  __rulesRef.current = rules
  toastError.mockClear()
  window.confirm = () => true
  mockFetch({ ok: true, body: { success: true } })
})

describe('แท็บ Auto-learn', () => {
  it('ลบกฎผ่าน API route ที่ใช้ service role', async () => {
    render(<AutoLearnTab />)

    fireEvent.click(await screen.findByTestId('delete-rule'))

    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    const [url, init] = fetchMock.mock.calls[0]
    expect(String(url)).toBe(`/api/keyword-rules/${rules[0].id}`)
    expect(String((init as RequestInit).method)).toBe('DELETE')
  })

  it('คงกฎไว้บนจอเมื่อ route บอกว่าไม่มีแถวไหนถูกลบ', async () => {
    mockFetch({ ok: false, status: 404, body: { success: false, error: 'ไม่พบกฎที่ต้องการลบ' } })
    render(<AutoLearnTab />)

    fireEvent.click(await screen.findByTestId('delete-rule'))

    await waitFor(() => expect(toastError).toHaveBeenCalled())
    expect(screen.getByText('เครื่องเขียน')).toBeInTheDocument()
  })
})
