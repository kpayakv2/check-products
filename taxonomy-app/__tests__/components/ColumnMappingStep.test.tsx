import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import ColumnMappingStep from '@/components/Import/ColumnMappingStep'

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>
  }
}))

// Mock lucide-react icons dynamically using Proxy
jest.mock('lucide-react', () => {
  return new Proxy({}, {
    get: (target, prop) => {
      return (props: any) => <div data-testid={`mock-icon-${String(prop)}`} {...props} />;
    }
  });
});

// Mock file with text() method
const createMockFile = (content: string) => {
  const blob = new Blob([content], { type: 'text/csv' })
  const file = new File([blob], 'test.csv', { type: 'text/csv' })
  ;(file as any).text = jest.fn().mockResolvedValue(content)
  return file
}

/** รอให้อ่านไฟล์เสร็จก่อน — ทุกอย่างที่เหลือเรนเดอร์หลังจากนี้ */
const waitForParsed = () =>
  waitFor(() => {
    expect(screen.getByText(/จับคู่คอลัมน์/)).toBeInTheDocument()
  })

/** ช่องเลือกคอลัมน์เรียงตาม SYSTEM_FIELDS โดยช่องแรกคือ "ชื่อสินค้า" ซึ่งเป็นช่องบังคับ */
const productNameSelect = () => screen.getAllByRole('combobox')[0]

describe('ColumnMappingStep', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('ขึ้นสถานะกำลังอ่านไฟล์ก่อนที่จะ parse เสร็จ', () => {
    const mockFile = createMockFile('product_name\nProduct 1')

    render(<ColumnMappingStep file={mockFile} onComplete={jest.fn()} />)

    expect(screen.getByText(/กำลังตรวจสอบโครงสร้างไฟล์/)).toBeInTheDocument()
  })

  it('แสดงหัวคอลัมน์และข้อมูลตัวอย่างจากไฟล์', async () => {
    const mockFile = createMockFile('product_name,category\nProduct 1,unique')

    render(<ColumnMappingStep file={mockFile} onComplete={jest.fn()} />)
    await waitForParsed()

    // หัวตารางเรนเดอร์เป็น "1. product_name" จึงต้องจับแบบ regex
    expect(screen.getByRole('columnheader', { name: /product_name/ })).toBeInTheDocument()
    expect(screen.getByRole('columnheader', { name: /category/ })).toBeInTheDocument()
    expect(screen.getByRole('cell', { name: 'Product 1' })).toBeInTheDocument()
  })

  it('เดาคอลัมน์ชื่อสินค้าให้อัตโนมัติเมื่อหัวคอลัมน์ตรงกับคำที่รู้จัก', async () => {
    const mockFile = createMockFile('product_name,price\nProduct 1,100')

    render(<ColumnMappingStep file={mockFile} onComplete={jest.fn()} />)
    await waitForParsed()

    expect(productNameSelect()).toHaveValue('product_name')
    expect(screen.getByText('พร้อมประมวลผล')).toBeInTheDocument()
  })

  it('เตือนเมื่อยังไม่ได้เลือกคอลัมน์ชื่อสินค้า', async () => {
    const mockFile = createMockFile('col1,col2\nvalue1,value2')

    render(<ColumnMappingStep file={mockFile} onComplete={jest.fn()} />)
    await waitForParsed()

    expect(screen.getByText('ต้องการข้อมูลเพิ่ม')).toBeInTheDocument()
    expect(screen.getByText(/กรุณาเลือกคอลัมน์ "ชื่อสินค้า"/)).toBeInTheDocument()
  })

  it('เลือกคอลัมน์เองได้เมื่อระบบเดาไม่ได้', async () => {
    const mockFile = createMockFile('col1,col2\nvalue1,value2')

    render(<ColumnMappingStep file={mockFile} onComplete={jest.fn()} />)
    await waitForParsed()

    expect(screen.getByText('ต้องการข้อมูลเพิ่ม')).toBeInTheDocument()

    fireEvent.change(productNameSelect(), { target: { value: 'col1' } })

    expect(productNameSelect()).toHaveValue('col1')
    expect(screen.getByText('พร้อมประมวลผล')).toBeInTheDocument()
  })

  it('ปุ่มไปต่อถูกปิดไว้จนกว่าจะเลือกคอลัมน์ชื่อสินค้า', async () => {
    const mockFile = createMockFile('col1,col2\nvalue1,value2')

    render(<ColumnMappingStep file={mockFile} onComplete={jest.fn()} />)
    await waitForParsed()

    const nextButton = screen.getByRole('button', { name: /รอเลือกชื่อสินค้า/ })
    expect(nextButton).toBeDisabled()
  })

  it('ส่ง mapping และข้อมูลที่ parse แล้วออกไปเมื่อกดไปต่อ', async () => {
    const onComplete = jest.fn()
    const mockFile = createMockFile('product_name,price\nProduct 1,100')

    render(<ColumnMappingStep file={mockFile} onComplete={onComplete} />)
    await waitForParsed()

    fireEvent.click(screen.getByRole('button', { name: /เริ่มประมวลผลวิเคราะห์/ }))

    expect(onComplete).toHaveBeenCalledTimes(1)
    const [mapping, preview] = onComplete.mock.calls[0]
    expect(mapping.product_name).toBe('product_name')
    expect(preview.headers).toEqual(['product_name', 'price'])
    expect(preview.rows).toHaveLength(1)
  })

  it('เรียก onBack เมื่อกดย้อนกลับ', async () => {
    const onBack = jest.fn()
    const mockFile = createMockFile('product_name\nProduct 1')

    render(<ColumnMappingStep file={mockFile} onComplete={jest.fn()} onBack={onBack} />)
    await waitForParsed()

    fireEvent.click(screen.getByRole('button', { name: /ย้อนกลับ/ }))

    expect(onBack).toHaveBeenCalled()
  })

  it('ไฟล์ที่มีแต่หัวคอลัมน์ไม่ทำให้พัง', async () => {
    const mockFile = createMockFile('product_name,category')

    render(<ColumnMappingStep file={mockFile} onComplete={jest.fn()} />)
    await waitForParsed()

    expect(screen.getByRole('columnheader', { name: /product_name/ })).toBeInTheDocument()
  })
})
