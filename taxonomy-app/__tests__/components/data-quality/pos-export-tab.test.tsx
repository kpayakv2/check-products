import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import PosExportTab from '@/components/data-quality/PosExportTab'

const previewRow = {
  itemGroup: 'ของใช้ในบ้าน',
  subGroup1: 'ตะกร้า/กระจาด',
  subGroup2: '',
  itemId: 'HH001',
  barcode: 'HH001',
  itemName: 'ตะกร้ากลม 9899',
  unitName: 'ใบ',
  price: 45,
  vat: 'V',
}

const EMPTY_ISSUES = {
  missingItemId: 0, missingBarcode: 0, missingItemName: 0,
  missingUnitName: 0, missingItemGroup: 0, missingPrice: 0,
}

const mockGet = (body: unknown) => {
  global.fetch = jest.fn().mockResolvedValue({
    ok: true,
    json: async () => body,
  }) as unknown as typeof fetch
}

afterEach(() => jest.resetAllMocks())

it('แสดงยอดรวมที่รอส่งออก', async () => {
  mockGet({ success: true, total: 221, issues: EMPTY_ISSUES, preview: [previewRow] })
  render(<PosExportTab />)

  await waitFor(() =>
    expect(screen.getByTestId('pos-export-total')).toHaveTextContent('221')
  )
})

it('เตือนเฉพาะช่องบังคับที่ยังขาดจริง', async () => {
  mockGet({
    success: true,
    total: 221,
    issues: { ...EMPTY_ISSUES, missingPrice: 221 },
    preview: [previewRow],
  })
  render(<PosExportTab />)

  const warning = await screen.findByTestId('pos-export-warning')
  expect(warning).toHaveTextContent('221')
  expect(warning).toHaveTextContent('ราคาขาย')
  // รหัสสินค้าถูกออกให้อัตโนมัติแล้ว ไม่ควรขึ้นเตือน
  expect(warning).not.toHaveTextContent('รหัสสินค้า')
})

it('เตือนได้หลายช่องพร้อมกัน', async () => {
  mockGet({
    success: true,
    total: 10,
    issues: { ...EMPTY_ISSUES, missingPrice: 10, missingUnitName: 3 },
    preview: [previewRow],
  })
  render(<PosExportTab />)

  const warning = await screen.findByTestId('pos-export-warning')
  expect(warning).toHaveTextContent('ราคาขาย')
  expect(warning).toHaveTextContent('ชื่อหน่วย')
})

it('ไม่เตือนเมื่อข้อมูลครบ', async () => {
  mockGet({ success: true, total: 5, issues: EMPTY_ISSUES, preview: [previewRow] })
  render(<PosExportTab />)

  await waitFor(() => expect(screen.getByTestId('pos-export-total')).toBeInTheDocument())
  expect(screen.queryByTestId('pos-export-warning')).not.toBeInTheDocument()
})

it('ปิดปุ่มโหลดเมื่อไม่มีอะไรให้ส่งออก', async () => {
  mockGet({ success: true, total: 0, issues: EMPTY_ISSUES, preview: [] })
  render(<PosExportTab />)

  expect(await screen.findByTestId('pos-export-empty')).toBeInTheDocument()
  expect(screen.getByTestId('pos-export-download')).toBeDisabled()
})

it('โหลดพรีวิวไม่สำเร็จ ต้องไม่บอกว่าคิวว่าง', async () => {
  global.fetch = jest.fn().mockRejectedValue(new Error('Failed to fetch')) as unknown as typeof fetch
  render(<PosExportTab />)

  expect(await screen.findByTestId('pos-export-load-failed')).toBeInTheDocument()
  // ติ๊กถูกเขียว "ส่งออกครบแล้ว" ขัดกับ error ที่อยู่เหนือมัน — ห้ามขึ้นตอนไม่รู้สถานะจริง
  expect(screen.queryByTestId('pos-export-empty')).not.toBeInTheDocument()
  expect(screen.getByText('Failed to fetch')).toBeInTheDocument()
})

it('ตัวเลขบนการ์ดสรุปมาจากไฟล์จริง ไม่ใช่ตัวเลขที่พรีวิวไว้ก่อนกด', async () => {
  // ลำดับคอลัมน์ตาม POS_EXPORT_HEADERS: กลุ่ม, กลุ่มย่อย1, รหัส, บาร์โค้ด, ชื่อ, หน่วย, ราคา, vat, กลุ่มย่อย2
  const line = (price: string) =>
    ['ของใช้ในบ้าน', 'ตะกร้า', 'HH001', 'HH001', 'ตะกร้ากลม', 'ใบ', price, 'V', ''].join('\t')
  const fileText = ['หัวตาราง', line('45'), line(''), line('10')].join('\r\n') + '\r\n'

  const fetchMock = jest.fn()
    .mockResolvedValueOnce({
      ok: true,
      // คิวตอนเปิดหน้าเหลือรายการเดียว แต่ระหว่างนั้นมีของเข้ามาเพิ่ม
      json: async () => ({ success: true, total: 1, issues: EMPTY_ISSUES, preview: [previewRow] }),
    })
    .mockResolvedValueOnce({
      ok: true,
      blob: async () => ({ text: async () => fileText }) as unknown as Blob,
    })
  global.fetch = fetchMock as unknown as typeof fetch
  global.URL.createObjectURL = jest.fn(() => 'blob:fake')
  global.URL.revokeObjectURL = jest.fn()

  render(<PosExportTab />)
  await waitFor(() => expect(screen.getByTestId('pos-export-total')).toBeInTheDocument())
  await userEvent.click(screen.getByTestId('pos-export-download'))
  await userEvent.click(await screen.findByTestId('pos-export-confirm'))

  const success = await screen.findByTestId('pos-export-success')
  expect(success).toHaveTextContent('ส่งออกเรียบร้อย 3 รายการ')
  // ราคาว่าง 1 แถวในไฟล์จริง ทั้งที่ตอนพรีวิวบอกว่าครบ
  expect(success).toHaveTextContent('กรอกราคาให้ 1 แถวที่ยังว่าง')
})

it('กดโหลดแล้วยิง POST ไปที่ /api/pos-export', async () => {
  const fetchMock = jest.fn()
    .mockResolvedValueOnce({
      ok: true,
      json: async () => ({ success: true, total: 1, issues: EMPTY_ISSUES, preview: [previewRow] }),
    })
    .mockResolvedValueOnce({
      ok: true,
      blob: async () => new Blob(['header\r\nrow\r\n'], { type: 'text/plain' }),
    })
  global.fetch = fetchMock as unknown as typeof fetch
  global.URL.createObjectURL = jest.fn(() => 'blob:fake')
  global.URL.revokeObjectURL = jest.fn()

  render(<PosExportTab />)
  await waitFor(() => expect(screen.getByTestId('pos-export-total')).toBeInTheDocument())

  await userEvent.click(screen.getByTestId('pos-export-download'))

  // ส่วนที่เพิ่มจากแผนเดิม: POST เขียนฐานข้อมูลถาวร (ออกรหัสสินค้า + ตีว่าส่งออกแล้ว)
  // ดีไซน์สุดท้ายจึงบังคับให้ยืนยันอีกชั้นก่อน ปุ่ม pos-export-download เปิดกล่องยืนยัน
  // ส่วนปุ่มที่ยิง POST จริงคือ pos-export-confirm — assertion ด้านล่างยังเป็นของเดิมทุกตัว
  await userEvent.click(await screen.findByTestId('pos-export-confirm'))

  await waitFor(() => {
    expect(fetchMock).toHaveBeenLastCalledWith(
      '/api/pos-export',
      expect.objectContaining({ method: 'POST' })
    )
  })
})
