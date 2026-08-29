/**
 * แท็บ "รอการอนุมัติ" ของหน้า /import ตายไปแล้ว — มันอ่านจาก
 * `suggestion_method = 'hybrid_ai_preview'` ซึ่งมีที่เขียนอยู่ที่เดียวคือ
 * /api/import/process ที่ไม่มี UI เรียกแล้ว (เป็นของ ProcessingStep ที่ถูกลบไป)
 * ในฐานข้อมูลจริงจึงมี 0 แถวตลอดกาล
 */
import { render, screen, waitFor } from '@testing-library/react'

jest.mock('@/components/Layout/Sidebar', () => ({ __esModule: true, default: () => null }))
jest.mock('@/components/Layout/Header', () => ({ __esModule: true, default: () => null }))
jest.mock('@/components/Import/WizardTab', () => ({ __esModule: true, default: () => <div>wizard</div> }))
jest.mock('@/components/Import/ImportHistory', () => ({
  __esModule: true,
  default: () => <div>ตารางประวัติ</div>
}))

import ImportPage from '@/app/import/page'

const fetchMock = jest.fn(async () => ({ ok: true, json: async () => ({}) }))

beforeEach(() => {
  fetchMock.mockClear()
  global.fetch = fetchMock as unknown as typeof fetch
})

describe('หน้า /import', () => {
  it('ไม่มีแท็บรอการอนุมัติที่ตายแล้ว', async () => {
    render(<ImportPage />)

    expect(screen.queryByTestId('pending-reviews-card')).not.toBeInTheDocument()
    expect(screen.queryByText(/Pending Reviews/i)).not.toBeInTheDocument()
  })

  it('ไม่เรียก API ที่ถูกลบไปแล้ว', async () => {
    render(<ImportPage />)

    await waitFor(() => expect(screen.getByTestId('import-title')).toBeInTheDocument())
    const calledUrls = fetchMock.mock.calls.map(call => String((call as unknown[])[0]))
    expect(calledUrls.some(url => url.includes('/api/import/pending'))).toBe(false)
  })

  it('ยังมีประวัติการนำเข้าให้ดู', async () => {
    render(<ImportPage />)

    expect(await screen.findByText('ตารางประวัติ')).toBeInTheDocument()
  })
})
