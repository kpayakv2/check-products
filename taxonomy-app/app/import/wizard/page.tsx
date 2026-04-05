'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { toast } from 'react-hot-toast'
import WizardLayout, { WizardStep } from '@/components/Import/WizardLayout'
import ColumnMappingStep, { ColumnMapping } from '@/components/Import/ColumnMappingStep'
import ProcessingStep, { ProcessedProduct } from '@/components/Import/ProcessingStep'
import StorageImport from '@/components/Import/StorageImport'
import ApprovalStep from '@/components/Import/ApprovalStep'
import Sidebar from '@/components/Layout/Sidebar'
import Header from '@/components/Layout/Header'
import { 
  UploadIcon,
  ColumnsIcon,
  CpuIcon,
  CheckSquareIcon,
  CheckCircleIcon,
  FolderIcon
} from 'lucide-react'
import type { ParsedCSV } from '@/utils/csv-parser'

const wizardSteps: WizardStep[] = [
  {
    id: 'upload',
    name: 'อัปโหลดไฟล์',
    description: 'เลือกไฟล์ CSV จาก Product Similarity Checker',
    icon: <UploadIcon />
  },
  {
    id: 'mapping',
    name: 'เลือกคอลัมน์',
    description: 'กำหนดว่าคอลัมน์ไหนคือชื่อสินค้า',
    icon: <ColumnsIcon />
  },
  {
    id: 'processing',
    name: 'ประมวลผล AI',
    description: 'AI วิเคราะห์และแนะนำหมวดหมู่',
    icon: <CpuIcon />
  },
  {
    id: 'review',
    name: 'ตรวจสอบ',
    description: 'ตรวจสอบและอนุมัติผลลัพธ์',
    icon: <CheckSquareIcon />
  },
  {
    id: 'complete',
    name: 'เสร็จสิ้น',
    description: 'สรุปผลและส่งออกข้อมูล',
    icon: <CheckCircleIcon />
  }
]

export default function ImportWizardPage() {
  const [currentStep, setCurrentStep] = useState(0)
  const [importMode, setImportMode] = useState<'upload' | 'storage'>('upload')
  const [file, setFile] = useState<File | null>(null)
  const [columnMapping, setColumnMapping] = useState<ColumnMapping | null>(null)
  const [parsedData, setParsedData] = useState<ParsedCSV | null>(null)
  const [processedProducts, setProcessedProducts] = useState<ProcessedProduct[]>([])

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
          <div className="space-y-6">
            {/* Import Mode Selection */}
            <div className="premium-card p-6">
              <h2 className="text-2xl font-bold mb-4 font-noto-sans-thai">
                📤 เลือกวิธีการ Import
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <button
                  onClick={() => setImportMode('upload')}
                  className={`p-6 border-2 rounded-lg transition-all ${
                    importMode === 'upload' 
                      ? 'border-blue-500 bg-blue-50' 
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <UploadIcon className="w-8 h-8 mx-auto mb-2 text-blue-500" />
                  <h3 className="font-semibold mb-2">อัปโหลดไฟล์ใหม่</h3>
                  <p className="text-sm text-gray-600">อัปโหลดไฟล์ CSV จากเครื่องของคุณ</p>
                </button>
                <button
                  onClick={() => setImportMode('storage')}
                  className={`p-6 border-2 rounded-lg transition-all ${
                    importMode === 'storage' 
                      ? 'border-blue-500 bg-blue-50' 
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <FolderIcon className="w-8 h-8 mx-auto mb-2 text-green-500" />
                  <h3 className="font-semibold mb-2">ใช้ไฟล์จาก Storage</h3>
                  <p className="text-sm text-gray-600">เลือกไฟล์ที่อัปโหลดไว้แล้ว</p>
                </button>
              </div>
            </div>

            {/* Upload Mode */}
            {importMode === 'upload' && (
              <div className="premium-card p-8">
                <h3 className="text-xl font-bold mb-4 font-noto-sans-thai">
                  📤 อัปโหลดไฟล์ CSV
                </h3>
                <p className="text-gray-600 mb-6">
                  เลือกไฟล์ <code className="bg-gray-100 px-2 py-1 rounded">approved_products_*.csv</code> 
                  {' '}จากระบบ Product Similarity Checker
                </p>

                <div className="border-2 border-dashed border-blue-300 rounded-lg p-12 text-center bg-blue-50 hover:bg-blue-100 transition-colors relative">
                <UploadIcon className="mx-auto h-16 w-16 text-blue-500 mb-4" />
                
                {file ? (
                  <div className="space-y-2">
                    <p className="text-lg font-semibold text-gray-900">{file.name}</p>
                    <p className="text-sm text-gray-500">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                    <button
                      onClick={() => setFile(null)}
                      className="text-sm text-red-600 hover:text-red-800"
                    >
                      ลบไฟล์
                    </button>
                  </div>
                ) : (
                  <div className="space-y-2">
                    <p className="text-lg text-gray-700">
                      ลากไฟล์มาวางที่นี่ หรือคลิกเพื่อเลือกไฟล์
                    </p>
                    <p className="text-sm text-gray-500">
                      รองรับไฟล์ .csv, .xlsx
                    </p>
                  </div>
                )}
                
                <input
                  type="file"
                  accept=".csv,.xlsx"
                  onChange={handleFileUpload}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                />
              </div>

              {file && (
                <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
                  <p className="text-green-800 text-sm">
                    ✅ ไฟล์พร้อมประมวลผล คลิก "ถัดไป" เพื่อดำเนินการต่อ
                  </p>
                </div>
              )}
              </div>
            )}

            {/* Storage Mode */}
            {importMode === 'storage' && (
              <div className="space-y-4">
                <StorageImport 
                  onFileSelect={(selectedFile, fileName) => {
                    setFile(selectedFile)
                    toast.success(`เลือกไฟล์: ${fileName}`)
                  }}
                />
                
                {file && (
                  <div className="premium-card p-4 bg-green-50 border border-green-200">
                    <p className="text-green-800 text-sm">
                      ✅ ไฟล์พร้อมประมวลผล: <strong>{file.name}</strong>
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Next Button */}
            {file && (
              <div className="flex justify-end mt-6">
                <button
                  onClick={handleNext}
                  className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors font-medium"
                >
                  ถัดไป: จับคู่คอลัมน์ →
                </button>
              </div>
            )}
          </div>
        )

      case 1:
        return file ? (
          <ColumnMappingStep
            file={file}
            onComplete={(mapping, preview) => {
              setColumnMapping(mapping)
              setParsedData(preview)
              handleNext()
            }}
            onBack={handleBack}
          />
        ) : (
          <div className="premium-card p-8">
            <p className="text-red-600">❌ กรุณาอัปโหลดไฟล์ก่อน</p>
            <button onClick={handleBack} className="btn-secondary mt-4">
              ← ย้อนกลับ
            </button>
          </div>
        )

      case 2:
        return file && columnMapping && parsedData ? (
          <ProcessingStep
            file={file}
            columnMapping={columnMapping}
            parsedData={parsedData}
            onComplete={(products) => {
              setProcessedProducts(products)
              handleNext()
            }}
            onBack={handleBack}
          />
        ) : (
          <div className="premium-card p-8">
            <p className="text-red-600">❌ ข้อมูลไม่ครบถ้วน กรุณาเริ่มต้นใหม่</p>
            <button onClick={() => setCurrentStep(0)} className="btn-secondary mt-4">
              ← เริ่มต้นใหม่
            </button>
          </div>
        )

      case 3:
        return (
          <ApprovalStep
            onComplete={(results) => {
              console.log('Approval completed:', results)
              handleNext()
            }}
            onBack={handleBack}
          />
        )

      case 4:
        const highConfidenceCount = processedProducts.filter(p => p.suggested_category.confidence_score >= 0.7).length
        const uniqueCategories = new Set(processedProducts.map(p => p.suggested_category.id)).size

        return (
          <div className="space-y-6">
            <div className="premium-card p-8 text-center">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", duration: 0.5 }}
              >
                <CheckCircleIcon className="mx-auto h-24 w-24 text-green-500 mb-6" />
              </motion.div>

              <h2 className="text-3xl font-bold mb-4 font-noto-sans-thai text-green-600">
                🎉 ประมวลผลเสร็จสิ้น!
              </h2>
              <p className="text-gray-600 mb-8">
                AI ได้วิเคราะห์และแนะนำหมวดหมู่สำหรับสินค้าทั้งหมดแล้ว
              </p>

              <div className="grid grid-cols-3 gap-4 mb-8">
                <div className="bg-blue-50 p-4 rounded-lg">
                  <p className="text-3xl font-bold text-blue-600">{processedProducts.length}</p>
                  <p className="text-sm text-gray-600">สินค้าทั้งหมด</p>
                </div>
                <div className="bg-green-50 p-4 rounded-lg">
                  <p className="text-3xl font-bold text-green-600">{highConfidenceCount}</p>
                  <p className="text-sm text-gray-600">ความมั่นใจสูง (≥70%)</p>
                </div>
                <div className="bg-purple-50 p-4 rounded-lg">
                  <p className="text-3xl font-bold text-purple-600">{uniqueCategories}</p>
                  <p className="text-sm text-gray-600">หมวดหมู่ที่แนะนำ</p>
                </div>
              </div>

              <div className="flex justify-center space-x-4">
                <button 
                  onClick={() => {
                    setCurrentStep(0)
                    setFile(null)
                    setColumnMapping(null)
                    setParsedData(null)
                    setProcessedProducts([])
                  }}
                  className="btn-secondary"
                >
                  นำเข้าไฟล์ใหม่
                </button>
                <button 
                  onClick={() => window.location.href = '/products'}
                  className="btn-premium"
                >
                  ดูสินค้าทั้งหมด
                </button>
              </div>
            </div>
          </div>
        )

      default:
        return null
    }
  }

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar />
      
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        
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
