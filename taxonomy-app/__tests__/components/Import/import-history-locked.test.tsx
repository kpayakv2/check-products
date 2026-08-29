/**
 * ตอนนี้ `/api/import/history` ถูก middleware กั้นแล้ว (มันอ่านผ่าน service role ข้าม RLS)
 * ถ้ายังไม่ปลดล็อกที่ /unlock จะได้ 401 กลับมา — เดิมโค้ดกลืน error ลง console แล้วโชว์
 * "ยังไม่มีประวัติ" ซึ่งเป็นความล้มเหลวเงียบแบบเดียวกับที่ไล่แก้มาทั้งชุด
 */
import { render, screen, waitFor } from '@testing-library/react'

jest.mock('@/utils/supabase', () => ({ supabase: { storage: { from: () => ({}) } } }))

import ImportHistory from '@/components/Import/ImportHistory'

const mockFetch = (status: number, body: unknown) => {
  global.fetch = jest.fn(async () => ({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body
  })) as unknown as typeof fetch
}

describe('ประวัติการนำเข้า', () => {
  it('บอกให้ไปปลดล็อกเมื่อยังไม่ได้ปลดล็อก แทนที่จะโชว์ว่าไม่มีประวัติ', async () => {
    mockFetch(401, { success: false, error: 'Unauthorized' })

    render(<ImportHistory />)

    const error = await screen.findByTestId('history-error')
    expect(error).toHaveTextContent('ปลดล็อก')
  })

  it('ไม่ขึ้นข้อความผิดพลาดเมื่อโหลดได้ปกติ', async () => {
    mockFetch(200, { success: true, data: [] })

    render(<ImportHistory />)

    await waitFor(() => expect(global.fetch).toHaveBeenCalled())
    expect(screen.queryByTestId('history-error')).not.toBeInTheDocument()
  })
})
