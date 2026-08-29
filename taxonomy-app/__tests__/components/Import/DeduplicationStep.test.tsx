import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import DeduplicationStep from '@/components/Import/DeduplicationStep'
import type { WizardItem } from '@/types/import'

jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>
  },
  AnimatePresence: ({ children }: any) => <>{children}</>
}))

jest.mock('lucide-react', () => new Proxy({}, {
  get: (_t, prop) => (props: any) => <div data-testid={`mock-icon-${String(prop)}`} {...props} />
}))

/**
 * โซนรีวิวคือรายการที่คะแนนก้ำกึ่ง 0.80-0.94 — ทำ backend ปลอมให้คืนคะแนนในช่วงนั้น
 * เพื่อให้ทุกตัวตกลงมาอยู่ในโซนที่ต้องให้คนตัดสิน
 */
const mockBackend = (pairs: unknown[]) => {
  global.fetch = jest.fn().mockResolvedValue({
    ok: true,
    json: async () => pairs
  }) as unknown as typeof fetch
}

const cleanedData: WizardItem[] = [
  { _cleaned_name: 'ปากกาลูกลื่น สีน้ำเงิน', price: 15 },
  { _cleaned_name: 'สมุดโน้ต A5', price: 45 }
]

const reviewPairs = [
  {
    newProduct: 'ปากกาลูกลื่น สีน้ำเงิน',
    oldProduct: 'ปากกาลูกลื่นน้ำเงิน',
    oldProductId: '11111111-1111-1111-1111-111111111111',
    oldPrice: 16,
    similarity: 0.88
  },
  {
    newProduct: 'สมุดโน้ต A5',
    oldProduct: 'สมุดโน้ต ขนาด A5',
    oldProductId: '22222222-2222-2222-2222-222222222222',
    oldPrice: 44,
    similarity: 0.85
  }
]

const renderStep = async (onComplete = jest.fn()) => {
  render(<DeduplicationStep cleanedData={cleanedData} onComplete={onComplete} onBack={jest.fn()} />)
  await waitFor(() => {
    expect(screen.getByText(/สินค้าชิ้นนี้ซ้ำกับของในสตอกหรือไม่/)).toBeInTheDocument()
  })
  return onComplete
}

/** อ่านผลของรายการหนึ่งจาก payload ที่ส่งออกไปตอนกดดำเนินการต่อ */
const bucketOf = (onComplete: jest.Mock, name: string) => {
  const deduped: WizardItem[] = onComplete.mock.calls[0][0]
  return deduped.find((item) => item._cleaned_name === name)?._bucket
}

const proceed = () => fireEvent.click(screen.getByRole('button', { name: /ดำเนินการต่อ/ }))

describe('DeduplicationStep — โซนรีวิวตัดสินได้จริง', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    mockBackend(reviewPairs)
  })

  it('ตรวจทีละรายการ และเลื่อนไปรายการถัดไปหลังตัดสิน', async () => {
    await renderStep()

    expect(screen.getByText('1 / 2')).toBeInTheDocument()
    expect(screen.getByText('ปากกาลูกลื่น สีน้ำเงิน')).toBeInTheDocument()

    fireEvent.keyDown(window, { key: 'a', code: 'KeyA' })

    expect(screen.getByText('2 / 2')).toBeInTheDocument()
    expect(screen.getByText('สมุดโน้ต A5')).toBeInTheDocument()
  })

  it('ปุ่ม A ตัดสินว่าซ้ำ และผลติดไปกับข้อมูลที่ส่งออก', async () => {
    const onComplete = await renderStep()

    fireEvent.keyDown(window, { key: 'a', code: 'KeyA' })
    proceed()

    expect(bucketOf(onComplete, 'ปากกาลูกลื่น สีน้ำเงิน')).toBe('duplicate')
  })

  it('ปุ่ม D ตัดสินว่าเป็นของใหม่', async () => {
    const onComplete = await renderStep()

    fireEvent.keyDown(window, { key: 'd', code: 'KeyD' })
    proceed()

    expect(bucketOf(onComplete, 'ปากกาลูกลื่น สีน้ำเงิน')).toBe('new')
  })

  it('ปุ่ม S ข้ามไว้ก่อน คงสถานะรอตรวจเพื่อไปทำต่อที่หน้า Verify', async () => {
    const onComplete = await renderStep()

    fireEvent.keyDown(window, { key: 's', code: 'KeyS' })
    proceed()

    expect(bucketOf(onComplete, 'ปากกาลูกลื่น สีน้ำเงิน')).toBe('review')
  })

  it('รองรับผังแป้นไทย — ฟ ก ห ทำงานเหมือน A D S', async () => {
    const onComplete = await renderStep()

    fireEvent.keyDown(window, { key: 'ฟ', code: 'KeyA' })
    fireEvent.keyDown(window, { key: 'ก', code: 'KeyD' })
    proceed()

    expect(bucketOf(onComplete, 'ปากกาลูกลื่น สีน้ำเงิน')).toBe('duplicate')
    expect(bucketOf(onComplete, 'สมุดโน้ต A5')).toBe('new')
  })

  it('ปุ่มลูกศรทำงานเหมือนกัน', async () => {
    const onComplete = await renderStep()

    fireEvent.keyDown(window, { key: 'ArrowLeft', code: 'ArrowLeft' })
    proceed()

    expect(bucketOf(onComplete, 'ปากกาลูกลื่น สีน้ำเงิน')).toBe('duplicate')
  })

  it('กดปุ่มบนหน้าจอแทนคีย์ลัดได้ผลเหมือนกัน', async () => {
    const onComplete = await renderStep()

    fireEvent.click(screen.getByRole('button', { name: /ซ้ำ — มีอยู่แล้วในสตอก/ }))
    proceed()

    expect(bucketOf(onComplete, 'ปากกาลูกลื่น สีน้ำเงิน')).toBe('duplicate')
  })

  it('คีย์ลัดไม่ทำงานขณะพิมพ์อยู่ในช่องกรอกข้อความ', async () => {
    const onComplete = await renderStep()

    const input = document.createElement('input')
    document.body.appendChild(input)
    input.focus()

    fireEvent.keyDown(window, { key: 'a', code: 'KeyA' })

    // ยังอยู่รายการเดิม ไม่ถูกตัดสิน
    expect(screen.getByText('1 / 2')).toBeInTheDocument()

    input.remove()
    proceed()
    expect(bucketOf(onComplete, 'ปากกาลูกลื่น สีน้ำเงิน')).toBe('review')
  })

  it('รายการที่ยังไม่ได้ตัดสินคงสถานะรอตรวจไว้ กดดำเนินการต่อได้เลย', async () => {
    const onComplete = await renderStep()

    proceed()

    expect(bucketOf(onComplete, 'ปากกาลูกลื่น สีน้ำเงิน')).toBe('review')
    expect(bucketOf(onComplete, 'สมุดโน้ต A5')).toBe('review')
  })

  it('ตัดสินครบทุกรายการแล้วขึ้นข้อความว่าตรวจครบ', async () => {
    await renderStep()

    fireEvent.keyDown(window, { key: 'a', code: 'KeyA' })
    fireEvent.keyDown(window, { key: 'd', code: 'KeyD' })

    expect(screen.getByText(/ตรวจครบทั้ง 2 รายการแล้ว/)).toBeInTheDocument()
  })
})
