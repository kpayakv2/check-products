/**
 * หน้านี้เคยเขียน DB ตรงด้วย anon key ซึ่ง RLS ปฏิเสธ — เพิ่มไม่ได้ แก้ไม่ได้
 * และ "ลบสำเร็จ" ทั้งที่ข้อมูลยังอยู่ ตอนนี้ทุกการเขียนต้องผ่าน API route
 * ที่ใช้ service role และต้องเชื่อผลลัพธ์จาก route จริง ๆ ไม่ใช่เดาว่าสำเร็จ
 */
import { render, screen, fireEvent, waitFor } from '@testing-library/react'

const tree = [
  {
    id: 'root-1',
    code: 'C1',
    name_th: 'เครื่องเขียน',
    level: 0,
    sort_order: 0,
    is_active: true,
    created_at: '2026-08-01T00:00:00Z',
    updated_at: '2026-08-01T00:00:00Z',
    children: [
      {
        id: 'child-1',
        code: 'C1-1',
        name_th: 'ปากกา',
        level: 1,
        sort_order: 0,
        is_active: true,
        created_at: '2026-08-01T00:00:00Z',
        updated_at: '2026-08-01T00:00:00Z',
        children: []
      },
      {
        id: 'child-2',
        code: 'C1-2',
        name_th: 'ดินสอ',
        level: 1,
        sort_order: 1,
        is_active: true,
        created_at: '2026-08-01T00:00:00Z',
        updated_at: '2026-08-01T00:00:00Z',
        children: []
      }
    ]
  }
]

jest.mock('@/components/Layout/Sidebar', () => ({ __esModule: true, default: () => null }))
jest.mock('@/components/Layout/Header', () => ({ __esModule: true, default: () => null }))
jest.mock('@/components/Taxonomy/SynonymsPanel', () => ({ __esModule: true, default: () => null }))

jest.mock('react-hot-toast', () => ({ toast: { error: jest.fn(), success: jest.fn() } }))

jest.mock('@/utils/supabase', () => ({
  DatabaseService: {
    getTaxonomyTree: jest.fn(async () => tree)
  }
}))

import { toast } from 'react-hot-toast'
import TaxonomyPage from '@/app/taxonomy/page'

const toastError = toast.error as jest.Mock
const toastSuccess = toast.success as jest.Mock

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
  toastError.mockClear()
  toastSuccess.mockClear()
  mockFetch({ ok: true, body: { success: true, data: { id: 'new-1' } } })
  window.confirm = () => true
})

const openCreateForm = async () => {
  fireEvent.click(await screen.findByRole('button', { name: /create node/i }))
}

describe('หน้าโครงสร้างหมวดหมู่', () => {
  it('นับจำนวนหมวดทั้งหมด ไม่ใช่เฉพาะหมวดบนสุด', async () => {
    render(<TaxonomyPage />)

    expect(await screen.findByTestId('node-count')).toHaveTextContent('3')
  })

  it('สร้างหมวดหมู่ผ่าน API route ที่ใช้ service role', async () => {
    render(<TaxonomyPage />)
    await openCreateForm()

    fireEvent.change(screen.getByPlaceholderText('เช่น อุปกรณ์เครื่องเขียน'), {
      target: { value: 'ของใช้ในบ้าน' }
    })
    fireEvent.click(screen.getByRole('button', { name: /commit node/i }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    const [url, init] = fetchMock.mock.calls[0]
    expect(String(url)).toBe('/api/taxonomy')
    expect(String((init as RequestInit).method)).toBe('POST')
  })

  it('ไม่ส่ง parent_id เป็นสตริงว่างไปกับคำขอ', async () => {
    render(<TaxonomyPage />)
    await openCreateForm()

    fireEvent.change(screen.getByPlaceholderText('เช่น อุปกรณ์เครื่องเขียน'), {
      target: { value: 'ของใช้ในบ้าน' }
    })
    fireEvent.click(screen.getByRole('button', { name: /commit node/i }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    const body = JSON.parse(String((fetchMock.mock.calls[0][1] as RequestInit).body))
    expect(body.parent_id).toBeUndefined()
  })

  it('ไม่ขึ้นว่าลบสำเร็จ เมื่อ route บอกว่าไม่มีแถวถูกลบ', async () => {
    mockFetch({ ok: false, status: 404, body: { success: false, error: 'ไม่พบหมวดหมู่ที่ต้องการลบ' } })
    render(<TaxonomyPage />)

    const deleteButtons = await screen.findAllByTestId('delete-node')
    fireEvent.click(deleteButtons[0])

    await waitFor(() => expect(toastError).toHaveBeenCalled())
    expect(toastSuccess).not.toHaveBeenCalled()
  })
})
