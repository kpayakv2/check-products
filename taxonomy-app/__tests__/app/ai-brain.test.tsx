/**
 * การ์ด "ความมั่นใจในการปัดตก" เคยโชว์ 85% เสมอ เพราะ /learn/status ส่งค่าคงที่กลับมา
 * ตอนนี้ backend ส่งค่าจริงจากตอนเทรน และส่ง null เมื่อโมเดลถูกเทรนไว้ก่อนจะมีการเก็บค่านี้
 * — หน้าเว็บต้องซ่อนการ์ดไปเลย ไม่ใช่แปลง null เป็น 0% หรือเดาเลขแทน
 */
import { render, screen, waitFor } from '@testing-library/react'

jest.mock('@/components/Layout/Sidebar', () => ({ __esModule: true, default: () => null }))
jest.mock('@/components/Layout/Header', () => ({ __esModule: true, default: () => null }))

import AiBrainPage from '@/app/ai-brain/page'

const mockApi = (status: Record<string, unknown>) => {
  global.fetch = jest.fn(async (url: unknown) => ({
    ok: true,
    json: async () =>
      String(url).includes('/status') ? status : { status: 'ok', history: [] }
  })) as unknown as typeof fetch
}

const TRAINED = {
  is_trained: true,
  accuracy: 99.64,
  total_samples: 1381,
  feature_importance: []
}

describe('หน้าสมองกล AI', () => {
  it('โชว์ความมั่นใจเฉลี่ยตามค่าที่โมเดลวัดได้จริง', async () => {
    mockApi({ ...TRAINED, average_confidence: 91.23 })

    render(<AiBrainPage />)

    expect(await screen.findByTestId('average-confidence')).toHaveTextContent('91.23%')
  })

  it('ซ่อนการ์ดไปเลยเมื่อโมเดลยังไม่มีค่าความมั่นใจให้รายงาน', async () => {
    mockApi({ ...TRAINED, average_confidence: null })

    render(<AiBrainPage />)

    await waitFor(() => expect(screen.getAllByText('1381').length).toBeGreaterThan(0))
    expect(screen.queryByTestId('average-confidence')).not.toBeInTheDocument()
    expect(screen.queryByText('85%')).not.toBeInTheDocument()
  })
})
