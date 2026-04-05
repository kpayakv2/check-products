'use client'

import { ReactNode } from 'react'
import { motion } from 'framer-motion'
import { 
  CheckCircleIcon, 
  CircleIcon,
  ArrowRightIcon 
} from 'lucide-react'

export interface WizardStep {
  id: string
  name: string
  description: string
  icon?: ReactNode
}

interface WizardLayoutProps {
  currentStep: number
  totalSteps: number
  steps: WizardStep[]
  children: ReactNode
  onStepClick?: (stepIndex: number) => void
  allowStepNavigation?: boolean
}

export default function WizardLayout({
  currentStep,
  totalSteps,
  steps,
  children,
  onStepClick,
  allowStepNavigation = false
}: WizardLayoutProps) {
  
  const getStepStatus = (stepIndex: number): 'completed' | 'current' | 'upcoming' => {
    if (stepIndex < currentStep) return 'completed'
    if (stepIndex === currentStep) return 'current'
    return 'upcoming'
  }

  const handleStepClick = (stepIndex: number) => {
    if (allowStepNavigation && onStepClick && stepIndex <= currentStep) {
      onStepClick(stepIndex)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Step Indicator */}
      <div className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <nav aria-label="Progress">
            <ol className="flex items-center justify-between">
              {steps.map((step, index) => {
                const status = getStepStatus(index)
                const isClickable = allowStepNavigation && index <= currentStep
                
                return (
                  <li 
                    key={step.id} 
                    className="relative flex-1"
                  >
                    {/* Connector Line */}
                    {index < steps.length - 1 && (
                      <div className="absolute top-5 left-1/2 w-full h-0.5 -z-10">
                        <div 
                          className={`h-full transition-colors duration-300 ${
                            status === 'completed' 
                              ? 'bg-blue-600' 
                              : 'bg-gray-200'
                          }`}
                        />
                      </div>
                    )}

                    {/* Step Button */}
                    <button
                      onClick={() => handleStepClick(index)}
                      disabled={!isClickable}
                      className={`relative flex flex-col items-center group ${
                        isClickable ? 'cursor-pointer' : 'cursor-default'
                      }`}
                      aria-current={status === 'current' ? 'step' : undefined}
                    >
                      {/* Step Circle */}
                      <motion.div
                        initial={false}
                        animate={{
                          scale: status === 'current' ? 1.1 : 1,
                          backgroundColor: 
                            status === 'completed' ? '#2563eb' :
                            status === 'current' ? '#3b82f6' :
                            '#e5e7eb'
                        }}
                        transition={{ duration: 0.3 }}
                        className={`
                          flex items-center justify-center w-10 h-10 rounded-full
                          border-2 transition-all duration-300
                          ${
                            status === 'completed' 
                              ? 'border-blue-600 bg-blue-600' 
                              : status === 'current'
                              ? 'border-blue-500 bg-blue-500'
                              : 'border-gray-300 bg-gray-100'
                          }
                          ${isClickable ? 'hover:shadow-lg' : ''}
                        `}
                      >
                        {status === 'completed' ? (
                          <CheckCircleIcon className="w-6 h-6 text-white" />
                        ) : status === 'current' ? (
                          <span className="text-white font-bold">{index + 1}</span>
                        ) : (
                          <CircleIcon className="w-5 h-5 text-gray-400" />
                        )}
                      </motion.div>

                      {/* Step Label */}
                      <div className="mt-3 text-center">
                        <motion.p
                          initial={false}
                          animate={{
                            color: 
                              status === 'completed' || status === 'current'
                                ? '#1e40af'
                                : '#6b7280',
                            fontWeight: status === 'current' ? 600 : 500
                          }}
                          className="text-sm font-noto-sans-thai"
                        >
                          {step.name}
                        </motion.p>
                        {status === 'current' && (
                          <motion.p
                            initial={{ opacity: 0, y: -5 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="text-xs text-gray-500 mt-1"
                          >
                            {step.description}
                          </motion.p>
                        )}
                      </div>

                      {/* Current Step Indicator */}
                      {status === 'current' && (
                        <motion.div
                          layoutId="currentStepIndicator"
                          className="absolute -bottom-2 left-1/2 transform -translate-x-1/2"
                          transition={{ type: "spring", stiffness: 300, damping: 30 }}
                        >
                          <ArrowRightIcon className="w-4 h-4 text-blue-600 rotate-90" />
                        </motion.div>
                      )}
                    </button>
                  </li>
                )
              })}
            </ol>
          </nav>

          {/* Progress Bar */}
          <div className="mt-6">
            <div className="flex items-center justify-between text-sm text-gray-600 mb-2">
              <span className="font-medium">
                ขั้นตอนที่ {currentStep + 1} จาก {totalSteps}
              </span>
              <span className="font-medium">
                {Math.round(((currentStep + 1) / totalSteps) * 100)}% เสร็จสมบูรณ์
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <motion.div
                initial={{ width: 0 }}
                animate={{ 
                  width: `${((currentStep + 1) / totalSteps) * 100}%` 
                }}
                transition={{ duration: 0.5, ease: "easeInOut" }}
                className="bg-gradient-to-r from-blue-500 to-blue-600 h-2 rounded-full"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Content Area */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        <motion.div
          key={currentStep}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.3 }}
        >
          {children}
        </motion.div>
      </div>
    </div>
  )
}
