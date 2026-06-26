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
    <div className="min-h-screen bg-slate-950 text-slate-200 relative overflow-hidden font-sans">
      {/* Decorative Dark Mode Background */}
      <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-purple-900/20 rounded-full blur-[150px] pointer-events-none -mr-48 -mt-48" />
      <div className="absolute bottom-0 left-0 w-[500px] h-[500px] bg-blue-900/20 rounded-full blur-[120px] pointer-events-none -ml-32 -mb-32" />

      {/* Step Indicator */}
      <div className="bg-slate-900/50 backdrop-blur-xl border-b border-white/5 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <nav aria-label="Progress">
            <ol className="flex items-center justify-between">
              {steps.map((step, index) => {
                const status = getStepStatus(index)
                const isClickable = allowStepNavigation && index <= currentStep
                
                return (
                  <li 
                    key={step.id} 
                    className="relative flex-1"
                    data-testid={`wizard-step-${step.id}`}
                  >
                    {/* Connector Line */}
                    {index < steps.length - 1 && (
                      <div className="absolute top-5 left-1/2 w-full h-0.5 -z-10">
                        <div 
                          className={`h-full transition-colors duration-300 ${
                            status === 'completed' 
                              ? 'bg-purple-600' 
                              : 'bg-slate-800'
                          }`}
                        />
                      </div>
                    )}

                    {/* Step Button */}
                    <button
                      onClick={() => handleStepClick(index)}
                      disabled={!isClickable}
                      data-testid={`wizard-step-button-${index}`}
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
                            status === 'completed' ? '#7c3aed' : // purple-600
                            status === 'current' ? '#8b5cf6' : // purple-500
                            '#1e293b' // slate-800
                        }}
                        transition={{ duration: 0.3 }}
                        className={`
                          flex items-center justify-center w-10 h-10 rounded-full
                          border-2 transition-all duration-300
                          ${
                            status === 'completed' 
                              ? 'border-purple-500 shadow-[0_0_15px_rgba(124,58,237,0.5)]' 
                              : status === 'current'
                              ? 'border-purple-400 shadow-[0_0_20px_rgba(139,92,246,0.6)]'
                              : 'border-slate-700 bg-slate-800'
                          }
                          ${isClickable ? 'hover:shadow-lg hover:shadow-purple-500/20' : ''}
                        `}
                      >
                        {status === 'completed' ? (
                          <CheckCircleIcon className="w-6 h-6 text-white" />
                        ) : status === 'current' ? (
                          <span className="text-white font-bold">{index + 1}</span>
                        ) : (
                          <CircleIcon className="w-5 h-5 text-slate-500" />
                        )}
                      </motion.div>

                      {/* Step Label */}
                      <div className="mt-4 text-center">
                        <motion.p
                          initial={false}
                          animate={{
                            color: 
                              status === 'completed'
                                ? '#c4b5fd' // purple-300
                                : status === 'current'
                                ? '#f8fafc' // slate-50
                                : '#64748b', // slate-500
                            fontWeight: status === 'current' ? 700 : 500
                          }}
                          className="text-sm thai-text tracking-wide"
                        >
                          {step.name}
                        </motion.p>
                        {status === 'current' && (
                          <motion.p
                            initial={{ opacity: 0, y: -5 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="text-xs text-purple-300/70 mt-1.5 thai-text"
                          >
                            {step.description}
                          </motion.p>
                        )}
                      </div>

                      {/* Current Step Indicator */}
                      {status === 'current' && (
                        <motion.div
                          layoutId="currentStepIndicator"
                          className="absolute -bottom-4 left-1/2 transform -translate-x-1/2"
                          transition={{ type: "spring", stiffness: 300, damping: 30 }}
                        >
                          <ArrowRightIcon className="w-4 h-4 text-purple-400 rotate-90" />
                        </motion.div>
                      )}
                    </button>
                  </li>
                )
              })}
            </ol>
          </nav>

          {/* Progress Bar */}
          <div className="mt-8 pt-6 border-t border-white/5">
            <div className="flex items-center justify-between text-xs text-slate-400 mb-3 uppercase tracking-widest font-black font-mono">
              <span>
                Step {currentStep + 1} of {totalSteps}
              </span>
              <span className="text-purple-400">
                {Math.round(((currentStep + 1) / totalSteps) * 100)}% Complete
              </span>
            </div>
            <div className="w-full bg-slate-800/50 rounded-full h-1.5 overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ 
                  width: `${((currentStep + 1) / totalSteps) * 100}%` 
                }}
                transition={{ duration: 0.5, ease: "easeInOut" }}
                className="bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 h-1.5 rounded-full"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Content Area */}
      <div className="max-w-7xl mx-auto px-6 py-10 relative z-10">
        <motion.div
          key={currentStep}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.4, ease: "easeOut" }}
        >
          {children}
        </motion.div>
      </div>
    </div>
  )
}
