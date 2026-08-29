import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import WizardTab from '@/components/Import/WizardTab'
import type { WizardItem } from '@/types/import'

jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
    p: ({ children, ...props }: any) => <p {...props}>{children}</p>
  },
  AnimatePresence: ({ children }: any) => <>{children}</>
}))

jest.mock('lucide-react', () => new Proxy({}, {
  get: (_t, prop) => (props: any) => <div data-testid={`mock-icon-${String(prop)}`} {...props} />
}))

jest.mock('@/components/Layout/Sidebar', () => () => null)
jest.mock('@/components/Layout/Header', () => () => null)
jest.mock('@/components/Import/StorageImport', () => () => null)

const mockCleaned: WizardItem[] = [{ _cleaned_name: 'ปากกาลูกลื่น', price: 15 }]
const mockDeduped: WizardItem[] = [{ _cleaned_name: 'ปากกาลูกลื่น', price: 15, _bucket: 'new', _source: 'backend' }]
const mockCategorized: WizardItem[] = [{
  _cleaned_name: 'ปากกาลูกลื่น',
  _bucket: 'new',
  _source: 'backend',
  _suggested_category: 'เครื่องเขียน',
  _suggested_category_id: '33333333-3333-3333-3333-333333333333',
  _confidence: 0.9
}]

// แทนแต่ละขั้นด้วยปุ่มเดียวที่ยิง onComplete — เทสต์นี้สนใจการประสานงานของ WizardTab
// ไม่ใช่ UI ภายในของแต่ละขั้น ซึ่งมีเทสต์ของตัวเองแยกอยู่แล้ว
//
// ตัวแปรที่ขึ้นต้นด้วย mock อ้างใน factory ได้ (jest อนุญาตไว้) และค่าถูกอ่านตอนคลิก
// ไม่ใช่ตอนสร้าง factory จึงพ้นช่วง TDZ ไปแล้วเสมอ
jest.mock('@/components/Import/UploadAndMappingStep', () => ({ onComplete }: any) => (
  <button onClick={() => onComplete({ product_name: 'name' }, { headers: ['name'], rows: [{ name: 'ปากกาลูกลื่น' }] })}>
    stub-upload
  </button>
))
jest.mock('@/components/Import/DataCleaningStep', () => ({ onComplete }: any) => (
  <button onClick={() => onComplete(mockCleaned)}>stub-clean</button>
))
jest.mock('@/components/Import/DeduplicationStep', () => ({ onComplete }: any) => (
  <button onClick={() => onComplete(mockDeduped)}>stub-dedup</button>
))
jest.mock('@/components/Import/CategorizationStep', () => ({ onComplete }: any) => (
  <button onClick={() => onComplete(mockCategorized)}>stub-categorize</button>
))
jest.mock('@/components/Import/CompleteStep', () => ({ saveResult, saveError, onReset }: any) => (
  <div>
    <div data-testid="save-batch">{saveResult?.import_batch_id ?? 'ไม่มี'}</div>
    <div data-testid="save-error">{saveError ?? 'ไม่มี'}</div>
    <button onClick={onReset}>stub-reset</button>
  </div>
))

const commitCalls = () =>
  (global.fetch as jest.Mock).mock.calls
    .filter(([url]) => url === '/api/import/commit')
    .map(([, init]) => JSON.parse(init.body))

const mockCommit = (overrides: Record<string, unknown> = {}) => {
  global.fetch = jest.fn().mockImplementation(async (_url: string, init: any) => {
    const body = JSON.parse(init.body)
    if (body.action === 'dedup') {
      return {
        ok: true,
        json: async () => ({
          success: true,
          import_batch_id: '44444444-4444-4444-4444-444444444444',
          saved: 1,
          counts: { pending_review_category: 1 },
          products: [{ id: '55555555-5555-5555-5555-555555555555', name_th: 'ปากกาลูกลื่น', status: 'pending_review_category' }],
          ...overrides
        })
      }
    }
    return { ok: true, json: async () => ({ success: true, updated: 1 }) }
  }) as unknown as typeof fetch
}

/** เดินจากขั้นแรกจนจบขั้นตรวจของซ้ำ (ซึ่งเป็นจุดที่บันทึกลงฐานข้อมูลครั้งแรก) */
const runToDedup = async () => {
  fireEvent.click(screen.getByText('stub-upload'))
  fireEvent.click(screen.getByText('stub-clean'))
  fireEvent.click(screen.getByText('stub-dedup'))
  await waitFor(() => expect(commitCalls().length).toBeGreaterThan(0))
}

describe('WizardTab — กันบันทึกซ้ำและรายงานผลตามจริง', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    mockCommit()
  })

  it('ส่ง wizard_run_id ไปกับการบันทึกทุกครั้ง', async () => {
    render(<WizardTab />)
    await runToDedup()

    const [dedupCall] = commitCalls()
    expect(dedupCall.action).toBe('dedup')
    expect(typeof dedupCall.wizard_run_id).toBe('string')
    expect(dedupCall.wizard_run_id.length).toBeGreaterThanOrEqual(8)
  })

  it('ย้อนกลับไปขั้นตรวจของซ้ำแล้วเดินหน้าใหม่ ยังใช้ wizard_run_id เดิม', async () => {
    render(<WizardTab />)
    await runToDedup()

    // แถบขั้นตอนให้กดย้อนกลับได้ — นี่คือเส้นทางที่เคยทำให้เกิด batch ซ้ำ
    fireEvent.click(screen.getByTestId('wizard-step-button-2'))
    fireEvent.click(screen.getByText('stub-dedup'))
    await waitFor(() => expect(commitCalls().filter((c) => c.action === 'dedup')).toHaveLength(2))

    const dedupCalls = commitCalls().filter((c) => c.action === 'dedup')
    expect(dedupCalls[1].wizard_run_id).toBe(dedupCalls[0].wizard_run_id)
  })

  it('กดเริ่มใหม่แล้วได้ wizard_run_id ใหม่ ไม่ไปทับ batch เดิม', async () => {
    render(<WizardTab />)
    await runToDedup()
    fireEvent.click(screen.getByText('stub-categorize'))
    await waitFor(() => expect(screen.getByTestId('save-batch')).toHaveTextContent('44444444'))

    fireEvent.click(screen.getByText('stub-reset'))
    await runToDedup()

    const dedupCalls = commitCalls().filter((c) => c.action === 'dedup')
    expect(dedupCalls).toHaveLength(2)
    expect(dedupCalls[1].wizard_run_id).not.toBe(dedupCalls[0].wizard_run_id)
  })

  it('กดเริ่มใหม่แล้วล้างผลบันทึกของรอบก่อนทิ้ง', async () => {
    render(<WizardTab />)
    await runToDedup()
    fireEvent.click(screen.getByText('stub-categorize'))
    await waitFor(() => expect(screen.getByTestId('save-batch')).toHaveTextContent('44444444'))

    fireEvent.click(screen.getByText('stub-reset'))

    // กลับมาถึงหน้าสรุปอีกครั้งโดยไม่ได้บันทึกอะไรเลย ต้องไม่เห็นผลของรอบก่อน
    fireEvent.click(screen.getByText('stub-upload'))
    fireEvent.click(screen.getByText('stub-clean'))
    fireEvent.click(screen.getByTestId('wizard-step-button-0'))
    expect(screen.queryByTestId('save-batch')).not.toBeInTheDocument()
  })

  it('บันทึกหมวดหมู่ไม่ได้เพราะยังไม่มี batch ต้องขึ้นข้อความ ไม่ใช่เงียบ', async () => {
    // ขั้นตรวจของซ้ำล้มเหลว จึงไม่มี import_batch_id ให้ขั้นจัดหมวดใช้ต่อ
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      json: async () => ({ success: false, error: 'backend ล่ม' })
    }) as unknown as typeof fetch

    render(<WizardTab />)
    fireEvent.click(screen.getByText('stub-upload'))
    fireEvent.click(screen.getByText('stub-clean'))
    fireEvent.click(screen.getByText('stub-dedup'))
    fireEvent.click(screen.getByText('stub-categorize'))

    await waitFor(() => {
      expect(screen.getByTestId('save-error')).toHaveTextContent('ยังไม่ได้บันทึกรายการจากขั้นตรวจของซ้ำ')
    })
  })

  it('จับคู่สินค้ากับรายการที่บันทึกไว้ไม่ได้ ต้องขึ้นข้อความ ไม่ใช่เงียบ', async () => {
    // ฝั่งเซิร์ฟเวอร์คืนชื่อที่ไม่ตรงกับข้อมูลในเบราว์เซอร์ จับคู่ไม่ได้สักตัว
    mockCommit({ products: [{ id: '66666666-6666-6666-6666-666666666666', name_th: 'ชื่อที่ไม่ตรงกัน', status: 'pending_review_category' }] })

    render(<WizardTab />)
    await runToDedup()
    fireEvent.click(screen.getByText('stub-categorize'))

    await waitFor(() => {
      expect(screen.getByTestId('save-error')).toHaveTextContent('จับคู่สินค้ากับรายการที่บันทึกไว้ไม่ได้')
    })
  })
})
