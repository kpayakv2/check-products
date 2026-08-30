/**
 * ปุ่ม "ออกจากระบบ" เดิมไม่มี onClick กดแล้วไม่เกิดอะไรขึ้นเลย
 * ตอนนี้ต้องเรียก POST /api/lock (ลบคุกกี้) แล้วพากลับไปหน้า /unlock จริง ๆ
 */
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import Header from '@/components/Layout/Header'

const push = jest.fn()
const refresh = jest.fn()

jest.mock('next/navigation', () => ({
  useRouter: () => ({ push, refresh }),
}))

describe('Header — ออกจากระบบ', () => {
  beforeEach(() => {
    push.mockClear()
    refresh.mockClear()
    global.fetch = jest.fn().mockResolvedValue({ ok: true, json: async () => ({ success: true }) })
  })

  it('เรียก POST /api/lock แล้วพากลับไปหน้า /unlock', async () => {
    const user = userEvent.setup()
    render(<Header />)

    await user.click(screen.getByText('ผู้ดูแลระบบ').closest('button')!)
    await user.click(await screen.findByText('ออกจากระบบ'))

    expect(global.fetch).toHaveBeenCalledWith('/api/lock', { method: 'POST' })
    expect(push).toHaveBeenCalledWith('/unlock')
    expect(refresh).toHaveBeenCalled()
  })
})
