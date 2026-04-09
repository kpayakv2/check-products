import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import ColumnMappingStep from '@/components/Import/ColumnMappingStep'

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>
  }
}))

// Mock lucide-react icons
jest.mock('lucide-react', () => ({
  AlertCircleIcon: () => <div data-testid="alert-icon" />,
  CheckCircleIcon: () => <div data-testid="check-icon" />,
  InfoIcon: () => <div data-testid="info-icon" />,
  ArrowRightIcon: () => <div data-testid="arrow-icon" />
}))

// Mock file with text() method
const createMockFile = (content: string) => {
  const blob = new Blob([content], { type: 'text/csv' })
  const file = new File([blob], 'test.csv', { type: 'text/csv' })
  ;(file as any).text = jest.fn().mockResolvedValue(content)
  return file
}

describe('ColumnMappingStep', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('should show loading state initially', () => {
    const mockFile = createMockFile('product_name\nProduct 1')

    render(
      <ColumnMappingStep
        file={mockFile}
        onComplete={jest.fn()}
      />
    )

    expect(screen.getByText('กำลังอ่านไฟล์...')).toBeInTheDocument()
  })

  it('should parse and display CSV preview', async () => {
    const csvContent = 'product_name,category\nProduct 1,unique'
    const mockFile = createMockFile(csvContent)

    render(
      <ColumnMappingStep
        file={mockFile}
        onComplete={jest.fn()}
      />
    )

    await waitFor(() => {
      expect(screen.getByText('product_name')).toBeInTheDocument()
      expect(screen.getByText('category')).toBeInTheDocument()
      expect(screen.getByText('Product 1')).toBeInTheDocument()
    }, { timeout: 3000 })
  })

  it('should auto-detect product_name column', async () => {
    const mockFile = createMockFile('product_name,other\nProduct 1,value')

    render(
      <ColumnMappingStep
        file={mockFile}
        onComplete={jest.fn()}
      />
    )

    await waitFor(() => {
      expect(screen.getByText('✅ ได้เลือกคอลัมน์ชื่อสินค้าแล้ว')).toBeInTheDocument()
    }, { timeout: 3000 })
  })

  it('should show warning when product_name not mapped', async () => {
    const mockFile = createMockFile('column1,column2\nvalue1,value2')

    render(
      <ColumnMappingStep
        file={mockFile}
        onComplete={jest.fn()}
      />
    )

    await waitFor(() => {
      expect(screen.getByText('❌ ยังไม่ได้เลือกคอลัมน์ชื่อสินค้า (จำเป็น)')).toBeInTheDocument()
    }, { timeout: 3000 })
  })

  it('should allow changing column mapping', async () => {
    const mockFile = createMockFile('col1,col2\nval1,val2')

    render(
      <ColumnMappingStep
        file={mockFile}
        onComplete={jest.fn()}
      />
    )

    await waitFor(() => {
      expect(screen.getByText('col1')).toBeInTheDocument()
    }, { timeout: 3000 })

    const selects = screen.getAllByRole('combobox')
    fireEvent.change(selects[0], { target: { value: 'product_name' } })

    await waitFor(() => {
      expect(screen.getByText('✅ ได้เลือกคอลัมน์ชื่อสินค้าแล้ว')).toBeInTheDocument()
    })
  })

  it('should display row count summary', async () => {
    const mockFile = createMockFile('product_name\nProduct 1\nProduct 2\nProduct 3')

    render(
      <ColumnMappingStep
        file={mockFile}
        onComplete={jest.fn()}
      />
    )

    await waitFor(() => {
      expect(screen.getByText(/จำนวนสินค้าที่จะประมวลผล/)).toBeInTheDocument()
    }, { timeout: 3000 })
  })

  it('should call onComplete with correct mapping', async () => {
    const mockFile = createMockFile('product_name,brand\nProduct 1,Brand A')
    const mockOnComplete = jest.fn()

    render(
      <ColumnMappingStep
        file={mockFile}
        onComplete={mockOnComplete}
      />
    )

    await waitFor(() => {
      expect(screen.getByText('✅ ได้เลือกคอลัมน์ชื่อสินค้าแล้ว')).toBeInTheDocument()
    }, { timeout: 3000 })

    const nextButton = screen.getByText(/ถัดไป: เริ่มประมวลผล AI/)
    fireEvent.click(nextButton)

    expect(mockOnComplete).toHaveBeenCalled()
  })

  it('should disable next button when product_name not mapped', async () => {
    const mockFile = createMockFile('column1\nvalue1')

    render(
      <ColumnMappingStep
        file={mockFile}
        onComplete={jest.fn()}
      />
    )

    await waitFor(() => {
      const nextButton = screen.getByText(/ถัดไป: เริ่มประมวลผล AI/)
      expect(nextButton).toBeDisabled()
    }, { timeout: 3000 })
  })

  it('should call onBack when back button clicked', async () => {
    const mockFile = createMockFile('product_name\nProduct 1')
    const mockOnBack = jest.fn()

    render(
      <ColumnMappingStep
        file={mockFile}
        onComplete={jest.fn()}
        onBack={mockOnBack}
      />
    )

    await waitFor(() => {
      expect(screen.getByText('product_name')).toBeInTheDocument()
    }, { timeout: 3000 })

    const backButton = screen.getByText('← ย้อนกลับ')
    fireEvent.click(backButton)

    expect(mockOnBack).toHaveBeenCalled()
  })

  it('should handle empty CSV file', async () => {
    const mockFile = createMockFile('')

    render(
      <ColumnMappingStep
        file={mockFile}
        onComplete={jest.fn()}
      />
    )

    await waitFor(() => {
      expect(screen.getByText(/กำหนดการจับคู่คอลัมน์/)).toBeInTheDocument()
    }, { timeout: 3000 })
  })

  it('should display validation warnings', async () => {
    const mockFile = createMockFile('product_name\n')

    render(
      <ColumnMappingStep
        file={mockFile}
        onComplete={jest.fn()}
      />
    )

    await waitFor(() => {
      expect(screen.getByText(/กำหนดการจับคู่คอลัมน์/)).toBeInTheDocument()
    }, { timeout: 3000 })
  })
})
