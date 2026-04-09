import { render, screen, waitFor } from '@testing-library/react'
import ProcessingStep from '@/components/Import/ProcessingStep'

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>
  },
  AnimatePresence: ({ children }: any) => <>{children}</>
}))

// Mock lucide-react
jest.mock('lucide-react', () => ({
  CheckCircleIcon: () => <div data-testid="check-icon" />,
  LoaderIcon: () => <div data-testid="loader-icon" />,
  AlertCircleIcon: () => <div data-testid="alert-icon" />,
  SparklesIcon: () => <div data-testid="sparkles-icon" />,
  BrainIcon: () => <div data-testid="brain-icon" />,
  ZapIcon: () => <div data-testid="zap-icon" />
}))

// Mock fetch
global.fetch = jest.fn()

// Mock TextEncoder for Node environment
if (typeof TextEncoder === 'undefined') {
  global.TextEncoder = class TextEncoder {
    encode(str: string) {
      return Buffer.from(str)
    }
  } as any
}

const mockFile = new File(['test content'], 'test.csv', { type: 'text/csv' })
const mockColumnMapping = {
  product_name: 'name',
  ignore: []
}
const mockParsedData = {
  headers: ['name'],
  rows: [{ name: 'Test Product' }],
  totalCount: 1
}

describe('ProcessingStep', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('should render processing step', () => {
    const mockOnComplete = jest.fn()

    render(
      <ProcessingStep
        file={mockFile}
        columnMapping={mockColumnMapping}
        parsedData={mockParsedData}
        onComplete={mockOnComplete}
      />
    )

    expect(screen.getByText('🤖 AI กำลังประมวลผล')).toBeInTheDocument()
    expect(screen.getByText(/กำลังวิเคราะห์สินค้า/)).toBeInTheDocument()
  })

  it('should display processing steps', () => {
    const mockOnComplete = jest.fn()

    render(
      <ProcessingStep
        file={mockFile}
        columnMapping={mockColumnMapping}
        parsedData={mockParsedData}
        onComplete={mockOnComplete}
      />
    )

    expect(screen.getByText('ทำความสะอาด')).toBeInTheDocument()
    expect(screen.getByText('แยกคำ')).toBeInTheDocument()
    expect(screen.getByText('สกัดคุณสมบัติ')).toBeInTheDocument()
    expect(screen.getByText('Vector Embeddings')).toBeInTheDocument()
    expect(screen.getByText('แนะนำหมวดหมู่')).toBeInTheDocument()
  })

  it('should show overall progress', () => {
    const mockOnComplete = jest.fn()

    render(
      <ProcessingStep
        file={mockFile}
        columnMapping={mockColumnMapping}
        parsedData={mockParsedData}
        onComplete={mockOnComplete}
      />
    )

    expect(screen.getByText('ความคืบหน้าโดยรวม')).toBeInTheDocument()
  })

  it('should display product count', () => {
    const mockOnComplete = jest.fn()

    render(
      <ProcessingStep
        file={mockFile}
        columnMapping={mockColumnMapping}
        parsedData={mockParsedData}
        onComplete={mockOnComplete}
      />
    )

    expect(screen.getByText(/0 \/ 1 สินค้า/)).toBeInTheDocument()
  })

  it('should render back button when onBack provided', () => {
    const mockOnComplete = jest.fn()
    const mockOnBack = jest.fn()

    render(
      <ProcessingStep
        file={mockFile}
        columnMapping={mockColumnMapping}
        parsedData={mockParsedData}
        onComplete={mockOnComplete}
        onBack={mockOnBack}
      />
    )

    const backButton = screen.getByText('← ย้อนกลับ')
    expect(backButton).toBeInTheDocument()
  })

  it('should have action buttons', () => {
    const mockOnComplete = jest.fn()

    render(
      <ProcessingStep
        file={mockFile}
        columnMapping={mockColumnMapping}
        parsedData={mockParsedData}
        onComplete={mockOnComplete}
      />
    )

    // Check if button exists (may show different text based on state)
    const buttons = screen.getAllByRole('button')
    expect(buttons.length).toBeGreaterThan(0)
  })
})
