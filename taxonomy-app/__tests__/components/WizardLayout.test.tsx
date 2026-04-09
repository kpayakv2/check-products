import { render, screen, fireEvent } from '@testing-library/react'
import WizardLayout, { WizardStep } from '@/components/Import/WizardLayout'

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
    p: ({ children, ...props }: any) => <p {...props}>{children}</p>
  }
}))

// Mock lucide-react icons
jest.mock('lucide-react', () => ({
  CheckCircleIcon: () => <div data-testid="check-icon" />,
  CircleIcon: () => <div data-testid="circle-icon" />,
  ArrowRightIcon: () => <div data-testid="arrow-icon" />
}))

const mockSteps: WizardStep[] = [
  {
    id: 'upload',
    name: 'อัปโหลดไฟล์',
    description: 'เลือกไฟล์ CSV ที่ต้องการนำเข้า'
  },
  {
    id: 'mapping',
    name: 'เลือกคอลัมน์',
    description: 'กำหนดว่าคอลัมน์ไหนคือชื่อสินค้า'
  },
  {
    id: 'processing',
    name: 'ประมวลผล',
    description: 'AI กำลังวิเคราะห์สินค้า'
  },
  {
    id: 'review',
    name: 'ตรวจสอบ',
    description: 'ตรวจสอบและอนุมัติผลลัพธ์'
  },
  {
    id: 'complete',
    name: 'เสร็จสิ้น',
    description: 'สรุปผลการนำเข้า'
  }
]

describe('WizardLayout', () => {
  it('should render all steps', () => {
    render(
      <WizardLayout
        currentStep={0}
        totalSteps={5}
        steps={mockSteps}
      >
        <div>Test Content</div>
      </WizardLayout>
    )

    // Check all step names are rendered
    expect(screen.getByText('อัปโหลดไฟล์')).toBeInTheDocument()
    expect(screen.getByText('เลือกคอลัมน์')).toBeInTheDocument()
    expect(screen.getByText('ประมวลผล')).toBeInTheDocument()
    expect(screen.getByText('ตรวจสอบ')).toBeInTheDocument()
    expect(screen.getByText('เสร็จสิ้น')).toBeInTheDocument()
  })

  it('should show current step description', () => {
    render(
      <WizardLayout
        currentStep={1}
        totalSteps={5}
        steps={mockSteps}
      >
        <div>Test Content</div>
      </WizardLayout>
    )

    // Current step (mapping) description should be visible
    expect(screen.getByText('กำหนดว่าคอลัมน์ไหนคือชื่อสินค้า')).toBeInTheDocument()
  })

  it('should display correct progress percentage', () => {
    render(
      <WizardLayout
        currentStep={2}
        totalSteps={5}
        steps={mockSteps}
      >
        <div>Test Content</div>
      </WizardLayout>
    )

    // Step 3 of 5 = 60%
    expect(screen.getByText('ขั้นตอนที่ 3 จาก 5')).toBeInTheDocument()
    expect(screen.getByText('60% เสร็จสมบูรณ์')).toBeInTheDocument()
  })

  it('should render children content', () => {
    render(
      <WizardLayout
        currentStep={0}
        totalSteps={5}
        steps={mockSteps}
      >
        <div data-testid="child-content">Test Content</div>
      </WizardLayout>
    )

    expect(screen.getByTestId('child-content')).toBeInTheDocument()
    expect(screen.getByText('Test Content')).toBeInTheDocument()
  })

  it('should call onStepClick when step is clicked (if allowed)', () => {
    const mockOnStepClick = jest.fn()

    render(
      <WizardLayout
        currentStep={2}
        totalSteps={5}
        steps={mockSteps}
        onStepClick={mockOnStepClick}
        allowStepNavigation={true}
      >
        <div>Test Content</div>
      </WizardLayout>
    )

    // Click on completed step (step 0)
    const stepButtons = screen.getAllByRole('button')
    fireEvent.click(stepButtons[0])

    expect(mockOnStepClick).toHaveBeenCalledWith(0)
  })

  it('should not call onStepClick when navigation is disabled', () => {
    const mockOnStepClick = jest.fn()

    render(
      <WizardLayout
        currentStep={2}
        totalSteps={5}
        steps={mockSteps}
        onStepClick={mockOnStepClick}
        allowStepNavigation={false}
      >
        <div>Test Content</div>
      </WizardLayout>
    )

    // Try to click on a step
    const stepButtons = screen.getAllByRole('button')
    fireEvent.click(stepButtons[0])

    expect(mockOnStepClick).not.toHaveBeenCalled()
  })

  it('should show step 1 as current step', () => {
    render(
      <WizardLayout
        currentStep={0}
        totalSteps={5}
        steps={mockSteps}
      >
        <div>Test Content</div>
      </WizardLayout>
    )

    // First step should have current step indicator
    const buttons = screen.getAllByRole('button')
    expect(buttons[0]).toHaveAttribute('aria-current', 'step')
  })

  it('should calculate progress correctly for first step', () => {
    render(
      <WizardLayout
        currentStep={0}
        totalSteps={5}
        steps={mockSteps}
      >
        <div>Test Content</div>
      </WizardLayout>
    )

    expect(screen.getByText('ขั้นตอนที่ 1 จาก 5')).toBeInTheDocument()
    expect(screen.getByText('20% เสร็จสมบูรณ์')).toBeInTheDocument()
  })

  it('should calculate progress correctly for last step', () => {
    render(
      <WizardLayout
        currentStep={4}
        totalSteps={5}
        steps={mockSteps}
      >
        <div>Test Content</div>
      </WizardLayout>
    )

    expect(screen.getByText('ขั้นตอนที่ 5 จาก 5')).toBeInTheDocument()
    expect(screen.getByText('100% เสร็จสมบูรณ์')).toBeInTheDocument()
  })

  it('should handle empty children', () => {
    render(
      <WizardLayout
        currentStep={0}
        totalSteps={5}
        steps={mockSteps}
      >
        {null}
      </WizardLayout>
    )

    // Should still render steps
    expect(screen.getByText('อัปโหลดไฟล์')).toBeInTheDocument()
  })

  it('should render with minimum required props', () => {
    const minimalSteps = [
      { id: 'step1', name: 'Step 1', description: 'Description 1' }
    ]

    render(
      <WizardLayout
        currentStep={0}
        totalSteps={1}
        steps={minimalSteps}
      >
        <div>Content</div>
      </WizardLayout>
    )

    expect(screen.getByText('Step 1')).toBeInTheDocument()
    expect(screen.getByText('100% เสร็จสมบูรณ์')).toBeInTheDocument()
  })
})
