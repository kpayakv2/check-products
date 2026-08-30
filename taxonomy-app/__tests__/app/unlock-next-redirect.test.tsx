/**
 * middleware.ts ตอนนี้เด้งหน้าเว็บที่ยังไม่ปลดล็อกไปที่ /unlock?next=<หน้าที่ตั้งใจไป>
 * เดิมหน้านี้ปลดล็อกสำเร็จแล้วพาไปหน้าแรกเสมอ ("/") ไม่สนว่าตั้งใจจะไปไหน
 * ตอนนี้ต้องอ่าน ?next แล้วพากลับไปที่นั่น — แต่ต้องกันไม่ให้ next พาออกนอกแอปได้
 * (เช่น next=https://evil.com หรือ next=//evil.com) ไม่งั้นเป็นช่องโหว่ open redirect
 */
import { act, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import UnlockPage from '@/app/unlock/page'

const push = jest.fn()
let searchParams = new URLSearchParams()

jest.mock('next/navigation', () => ({
  useRouter: () => ({ push, refresh: jest.fn() }),
  useSearchParams: () => searchParams,
}))

async function submit(secret = 'ok-secret') {
  const user = userEvent.setup()
  render(<UnlockPage />)
  await user.type(screen.getByPlaceholderText('รหัสผ่าน'), secret)
  await user.click(screen.getByRole('button', { name: /ปลดล็อก/ }))
  // setSubmitting(false) ใน finally ยิงอีกรอบหลัง push แล้ว — รอให้ค้างจบก่อนตรวจผล
  await act(async () => { await Promise.resolve() })
}

describe('UnlockPage — พากลับไปหน้าที่ตั้งใจไปหลังปลดล็อก', () => {
  beforeEach(() => {
    push.mockClear()
    searchParams = new URLSearchParams()
    global.fetch = jest.fn().mockResolvedValue({ ok: true, json: async () => ({ success: true }) })
  })

  it('ไม่มี next → กลับไปหน้าแรกเหมือนเดิม', async () => {
    await submit()
    expect(push).toHaveBeenCalledWith('/')
  })

  it('มี next เป็น path ในแอป → กลับไปหน้านั้น', async () => {
    searchParams = new URLSearchParams('next=/data-quality')
    await submit()
    expect(push).toHaveBeenCalledWith('/data-quality')
  })

  it('next เป็น URL ภายนอก → ไม่พาออกนอกแอป กลับไปหน้าแรกแทน', async () => {
    searchParams = new URLSearchParams('next=https://evil.com')
    await submit()
    expect(push).toHaveBeenCalledWith('/')
  })

  it('next เป็น protocol-relative URL → ไม่พาออกนอกแอป กลับไปหน้าแรกแทน', async () => {
    searchParams = new URLSearchParams('next=' + encodeURIComponent('//evil.com'))
    await submit()
    expect(push).toHaveBeenCalledWith('/')
  })
})
