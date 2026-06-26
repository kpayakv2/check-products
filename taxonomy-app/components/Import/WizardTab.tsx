'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { toast } from 'react-hot-toast'
import WizardLayout, { WizardStep } from '@/components/Import/WizardLayout'
import StorageImport from '@/components/Import/StorageImport'
import UploadAndMappingStep from '@/components/Import/UploadAndMappingStep'
import DataCleaningStep from '@/components/Import/DataCleaningStep'
import DeduplicationStep from '@/components/Import/DeduplicationStep'
import CategorizationStep from '@/components/Import/CategorizationStep'
import CompleteStep from '@/components/Import/CompleteStep'
import Sidebar from '@/components/Layout/Sidebar'
import Header from '@/components/Layout/Header'
import { 
  UploadIcon,
  ColumnsIcon,
  CpuIcon,
  CheckSquareIcon,
  CheckCircleIcon,
  FolderIcon,
  SparklesIcon,
  CopyIcon,
  TagsIcon,
  DatabaseIcon,
  ArrowRightIcon
} from 'lucide-react'
import type { ParsedCSV } from '@/utils/csv-parser'

const wizardSteps: WizardStep[] = [
  {
    id: 'upload',
    name: 'อัปโหลดข้อมูล',
    description: 'อัปโหลดและจับคู่คอลัมน์',
    icon: <UploadIcon />
  },
  {
    id: 'clean',
    name: 'ทำความสะอาด',
    description: 'พรีวิวข้อมูลที่ล้างแล้ว',
    icon: <SparklesIcon />
  },
  {
    id: 'dedup',
    name: 'จัดการของซ้ำ',
    description: 'ตรวจสอบสินค้าที่ชื่อคล้าย',
    icon: <CopyIcon />
  },
  {
    id: 'categorize',
    name: 'จัดหมวดหมู่',
    description: 'รีวิวหมวดหมู่แนะนำ',
    icon: <TagsIcon />
  },
  {
    id: 'complete',
    name: 'บันทึก',
    description: 'บันทึกและอัปเดต Knowledge Base',
    icon: <DatabaseIcon />
  }
]

export default function WizardTab() {
  const [currentStep, setCurrentStep] = useState(0)
  const [importMode, setImportMode] = useState<'upload' | 'storage'>('upload')
  const [file, setFile] = useState<File | null>(null)
  const [columnMapping, setColumnMapping] = useState<any | null>(null)
  
  // Pipeline State
  const [parsedData, setParsedData] = useState<ParsedCSV | null>(null)
  const [cleanedData, setCleanedData] = useState<any[]>([])
  const [dedupedData, setDedupedData] = useState<any[]>([])
  const [categorizedData, setCategorizedData] = useState<any[]>([])

  const handleNext = () => {
    if (currentStep < wizardSteps.length - 1) {
      setCurrentStep(currentStep + 1)
    }
  }

  const handleBack = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1)
    }
  }

  const handleStepClick = (stepIndex: number) => {
    // Allow going back to previous steps
    if (stepIndex <= currentStep) {
      setCurrentStep(stepIndex)
    }
  }

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = event.target.files?.[0]
    if (uploadedFile) {
      setFile(uploadedFile)
    }
  }

    const renderStepContent = () => {
      switch (currentStep) {
        case 0:
          return (
            <UploadAndMappingStep 
              file={file}
              setFile={setFile}
              importMode={importMode}
              setImportMode={setImportMode}
              onComplete={(mapping: any, preview: any) => {
                setColumnMapping(mapping)
                setParsedData(preview)
                handleNext()
              }}
            />
          )
  
        case 1:
          return parsedData && columnMapping ? (
            <DataCleaningStep
              parsedData={parsedData}
              columnMapping={columnMapping}
              onComplete={(cleaned: any) => {
                setCleanedData(cleaned)
                handleNext()
              }}
              onBack={handleBack}
            />
          ) : (
            <div className="p-8 text-center bg-red-50 text-red-600 rounded-2xl">
              <p className="font-bold text-xl mb-4">❌ กรุณาอัปโหลดไฟล์ก่อน</p>
              <button onClick={handleBack} className="px-6 py-2 bg-slate-200 text-slate-800 rounded-full">ย้อนกลับ</button>
            </div>
          )
  
        case 2:
          return cleanedData.length > 0 ? (
            <DeduplicationStep
              cleanedData={cleanedData}
              onComplete={(deduped: any) => {
                setDedupedData(deduped)
                handleNext()
              }}
              onBack={handleBack}
            />
          ) : (
            <div className="p-8 text-center bg-red-50 text-red-600 rounded-2xl">
              <p className="font-bold text-xl mb-4">❌ ไม่มีข้อมูลที่ถูกทำความสะอาด</p>
              <button onClick={handleBack} className="px-6 py-2 bg-slate-200 text-slate-800 rounded-full">ย้อนกลับ</button>
            </div>
          )
  
        case 3:
          return dedupedData.length > 0 ? (
            <CategorizationStep
              dedupedData={dedupedData}
              onComplete={(categorized: any) => {
                setCategorizedData(categorized)
                handleNext()
              }}
              onBack={handleBack}
            />
          ) : (
            <div className="p-8 text-center bg-red-50 text-red-600 rounded-2xl">
              <p className="font-bold text-xl mb-4">❌ ไม่มีข้อมูลที่ผ่านการตรวจสอบของซ้ำ</p>
              <button onClick={handleBack} className="px-6 py-2 bg-slate-200 text-slate-800 rounded-full">ย้อนกลับ</button>
            </div>
          )
  
        case 4:
          return (
            <CompleteStep 
              categorizedData={categorizedData} 
              onReset={() => {
                setCurrentStep(0)
                setFile(null)
                setColumnMapping(null)
                setParsedData(null)
                setCleanedData([])
                setDedupedData([])
                setCategorizedData([])
              }} 
            />
          )
  
        default:
          return null
      }
    }

  return (
    <div className="h-full bg-gray-50">
      <div className="flex-1 flex flex-col overflow-hidden">
        <main className="flex-1 overflow-y-auto">
          <WizardLayout
            currentStep={currentStep}
            totalSteps={wizardSteps.length}
            steps={wizardSteps}
            onStepClick={handleStepClick}
            allowStepNavigation={true}
          >
            {renderStepContent()}
          </WizardLayout>
        </main>
      </div>
    </div>
  )
}
