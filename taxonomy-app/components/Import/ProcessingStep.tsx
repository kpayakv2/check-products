'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { toast } from 'react-hot-toast'
import { supabase, DatabaseService } from '@/utils/supabase'
import {
  CheckCircleIcon,
  LoaderIcon,
  AlertCircleIcon,
  SparklesIcon,
  BrainIcon,
  ZapIcon
} from 'lucide-react'
import type { ParsedCSV } from '@/utils/csv-parser'
import type { ColumnMapping } from './ColumnMappingStep'

interface ProcessingStepProps {
  file: File
  columnMapping: ColumnMapping
  parsedData: ParsedCSV
  onComplete: (results: ProcessedProduct[]) => void
  onBack?: () => void
}

export interface ProcessedProduct {
  id: string
  name_th: string
  cleaned_name: string
  tokens: string[]
  units: string[]
  attributes: Record<string, any>
  embedding: number[]
  suggested_category: {
    id: string
    name_th: string
    confidence_score: number
    explanation: string
  }
  status: 'pending' | 'approved' | 'rejected'
}

interface ProcessingStep {
  id: string
  name: string
  icon: React.ReactNode
  status: 'pending' | 'processing' | 'completed' | 'error'
  progress: number
  message: string
}

const PROCESSING_STEPS: ProcessingStep[] = [
  {
    id: 'clean',
    name: 'ทำความสะอาด',
    icon: <SparklesIcon className="w-5 h-5" />,
    status: 'pending',
    progress: 0,
    message: 'รอดำเนินการ...'
  },
  {
    id: 'tokenize',
    name: 'แยกคำ',
    icon: <ZapIcon className="w-5 h-5" />,
    status: 'pending',
    progress: 0,
    message: 'รอดำเนินการ...'
  },
  {
    id: 'extract',
    name: 'สกัดคุณสมบัติ',
    icon: <BrainIcon className="w-5 h-5" />,
    status: 'pending',
    progress: 0,
    message: 'รอดำเนินการ...'
  },
  {
    id: 'embed',
    name: 'Vector Embeddings',
    icon: <BrainIcon className="w-5 h-5" />,
    status: 'pending',
    progress: 0,
    message: 'รอดำเนินการ...'
  },
  {
    id: 'suggest',
    name: 'แนะนำหมวดหมู่',
    icon: <CheckCircleIcon className="w-5 h-5" />,
    status: 'pending',
    progress: 0,
    message: 'รอดำเนินการ...'
  }
]

export default function ProcessingStep({
  file,
  columnMapping,
  parsedData,
  onComplete,
  onBack
}: ProcessingStepProps) {
  const [steps, setSteps] = useState<ProcessingStep[]>(PROCESSING_STEPS)
  const [isProcessing, setIsProcessing] = useState(false)
  const [processedProducts, setProcessedProducts] = useState<ProcessedProduct[]>([])
  const [error, setError] = useState<string | null>(null)
  const [currentProductIndex, setCurrentProductIndex] = useState(0)
  const [uploadedFilePath, setUploadedFilePath] = useState<string | null>(null)
  const [importBatchId, setImportBatchId] = useState<string | null>(null)

  useEffect(() => {
    if (!isProcessing) {
      startProcessing()
    }
  }, [])

  const startProcessing = async () => {
    setIsProcessing(true)
    setError(null)

    try {
      // Step 1: Upload file to Supabase Storage
      updateStep('clean', 'processing', 10, 'กำลังอัปโหลดไฟล์...')
      
      const fileName = `products/${Date.now()}_${file.name}`
      const { data: uploadData, error: uploadError } = await supabase.storage
        .from('uploads')
        .upload(fileName, file, {
          cacheControl: '3600',
          upsert: false
        })

      if (uploadError) {
        throw new Error(`Upload failed: ${uploadError.message}`)
      }

      setUploadedFilePath(uploadData.path)
      updateStep('clean', 'completed', 100, 'อัปโหลดไฟล์สำเร็จ')

      // Step 2: Create import batch record
      const importBatch = await DatabaseService.createImport({
        name: `Product Import - ${new Date().toLocaleString('th-TH')}`,
        description: `นำเข้าจากไฟล์ ${file.name}`,
        total_records: parsedData.totalCount,
        processed_records: 0,
        success_records: 0,
        error_records: 0,
        status: 'processing'
      })

      setImportBatchId(importBatch.id)
      toast.success(`สร้าง batch #${importBatch.id} สำเร็จ`)

      // Step 3: Process with AI
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch('/api/import/process', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error('Failed to process file')
      }

      const reader = response.body?.getReader()
      if (!reader) {
        throw new Error('No response stream')
      }

      const decoder = new TextDecoder()
      const products: ProcessedProduct[] = []
      let buffer = '' // Buffer for incomplete JSON

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        buffer += chunk
        
        // Split by newlines but keep incomplete lines in buffer
        const lines = buffer.split('\n')
        buffer = lines.pop() || '' // Keep last incomplete line in buffer

        for (const line of lines) {
          if (!line.trim()) continue
          
          try {
            const data = JSON.parse(line)

            if (data.type === 'step_update') {
              updateStep(data.step, data.status, data.progress, data.message)
            } else if (data.type === 'suggestion') {
              products.push(data.suggestion)
              setProcessedProducts([...products])
              setCurrentProductIndex(products.length - 1)
            } else if (data.type === 'error') {
              setError(data.message)
            }
          } catch (e) {
            console.error('Failed to parse line:', line.substring(0, 200), e)
          }
        }
      }
      
      // Process any remaining data in buffer
      if (buffer.trim()) {
        try {
          const data = JSON.parse(buffer)
          if (data.type === 'suggestion') {
            products.push(data.suggestion)
            setProcessedProducts([...products])
          }
        } catch (e) {
          console.error('Failed to parse final buffer:', buffer.substring(0, 200), e)
        }
      }

      // Step 4: Save processed products to database
      if (products.length > 0 && importBatchId) {
        updateStep('suggest', 'processing', 90, 'กำลังบันทึกลงฐานข้อมูล...')
        
        await saveProductsToDatabase(products, importBatchId)
        
        updateStep('suggest', 'completed', 100, `บันทึก ${products.length} สินค้าสำเร็จ`)
        toast.success(`นำเข้า ${products.length} สินค้าเรียบร้อย`)
      }

      setIsProcessing(false)
      onComplete(products)

    } catch (err) {
      console.error('Processing error:', err)
      setError(err instanceof Error ? err.message : 'เกิดข้อผิดพลาด')
      toast.error(`เกิดข้อผิดพลาด: ${err instanceof Error ? err.message : 'Unknown'}`)
      
      // Update import batch status to failed
      if (importBatchId) {
        await DatabaseService.updateImport(importBatchId, {
          status: 'failed',
          error_details: { 
            error: err instanceof Error ? err.message : 'Unknown error' 
          }
        })
      }
      
      setIsProcessing(false)
    }
  }

  const saveProductsToDatabase = async (products: ProcessedProduct[], batchId: string) => {
    const results = {
      success: 0,
      failed: 0,
      errors: [] as string[]
    }

    for (const product of products) {
      try {
        // Create product record
        const createdProduct = await DatabaseService.createProduct({
          name_th: product.name_th,
          description: product.cleaned_name,
          category_id: product.suggested_category.id || undefined,
          keywords: product.tokens,
          embedding: product.embedding,
          metadata: {
            units: product.units,
            attributes: product.attributes,
            original_text: product.name_th,
            cleaned_text: product.cleaned_name
          },
          status: 'pending',
          confidence_score: product.suggested_category.confidence_score,
          import_batch_id: batchId
        })

        // Create category suggestion record
        if (product.suggested_category.id) {
          await DatabaseService.createProductCategorySuggestion({
            product_id: createdProduct.id,
            suggested_category_id: product.suggested_category.id,
            confidence_score: product.suggested_category.confidence_score,
            suggestion_method: 'keyword_rule',
            metadata: {
              explanation: product.suggested_category.explanation,
              matched_tokens: product.tokens,
              processing_timestamp: new Date().toISOString()
            },
            is_accepted: false
          })
        }

        // Create product attributes
        if (product.attributes) {
          for (const [key, value] of Object.entries(product.attributes)) {
            if (Array.isArray(value)) {
              for (const item of value) {
                await DatabaseService.createProductAttribute({
                  product_id: createdProduct.id,
                  attribute_name: key,
                  attribute_value: String(item),
                  attribute_type: 'text'
                })
              }
            } else {
              await DatabaseService.createProductAttribute({
                product_id: createdProduct.id,
                attribute_name: key,
                attribute_value: String(value),
                attribute_type: typeof value === 'number' ? 'number' : 'text'
              })
            }
          }
        }

        results.success++
      } catch (error) {
        console.error(`Failed to create product: ${product.name_th}`, error)
        results.failed++
        results.errors.push(`${product.name_th}: ${error instanceof Error ? error.message : 'Unknown'}`)
      }
    }

    // Update import batch status
    await DatabaseService.updateImport(batchId, {
      processed_records: products.length,
      success_records: results.success,
      error_records: results.failed,
      status: results.failed === 0 ? 'completed' : 'failed',
      error_details: results.errors.length > 0 ? { errors: results.errors } : undefined,
      completed_at: new Date().toISOString()
    })

    if (results.failed > 0) {
      toast.error(`${results.failed} สินค้าไม่สามารถบันทึกได้`)
    }
  }

  const updateStep = (
    stepId: string,
    status: 'pending' | 'processing' | 'completed' | 'error',
    progress: number,
    message: string
  ) => {
    setSteps(prev => prev.map(step => 
      step.id === stepId
        ? { ...step, status, progress, message }
        : step
    ))
  }

  const getStepColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'text-green-600 bg-green-50 border-green-200'
      case 'processing':
        return 'text-blue-600 bg-blue-50 border-blue-200'
      case 'error':
        return 'text-red-600 bg-red-50 border-red-200'
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200'
    }
  }

  const getStepIcon = (step: ProcessingStep) => {
    if (step.status === 'completed') {
      return <CheckCircleIcon className="w-5 h-5 text-green-600" />
    } else if (step.status === 'processing') {
      return (
        <LoaderIcon className="w-5 h-5 text-blue-600 animate-spin" />
      )
    } else if (step.status === 'error') {
      return <AlertCircleIcon className="w-5 h-5 text-red-600" />
    }
    return step.icon
  }

  const overallProgress = Math.round(
    steps.reduce((sum, step) => sum + step.progress, 0) / steps.length
  )

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="premium-card p-6">
        <h2 className="text-2xl font-bold mb-2 font-noto-sans-thai">
          🤖 AI กำลังประมวลผล
        </h2>
        <p className="text-gray-600">
          กำลังวิเคราะห์สินค้าและแนะนำหมวดหมู่...
        </p>
      </div>

      {/* Overall Progress */}
      <div className="premium-card p-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700">
            ความคืบหน้าโดยรวม
          </span>
          <span className="text-sm font-bold text-blue-600">
            {overallProgress}%
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-blue-500 to-purple-500"
            initial={{ width: 0 }}
            animate={{ width: `${overallProgress}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
        <div className="mt-2 text-xs text-gray-500">
          {processedProducts.length} / {parsedData.totalCount} สินค้า
        </div>
      </div>

      {/* Processing Steps */}
      <div className="premium-card p-6">
        <h3 className="font-semibold mb-4">ขั้นตอนการประมวลผล:</h3>
        <div className="space-y-3">
          {steps.map((step, index) => (
            <motion.div
              key={step.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className={`
                border rounded-lg p-4 transition-all
                ${getStepColor(step.status)}
              `}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-3">
                  {getStepIcon(step)}
                  <span className="font-medium">{step.name}</span>
                </div>
                <span className="text-xs font-mono">
                  {step.progress}%
                </span>
              </div>
              
              {step.status !== 'pending' && (
                <>
                  <div className="w-full bg-white bg-opacity-50 rounded-full h-2 mb-2">
                    <motion.div
                      className={`h-full rounded-full ${
                        step.status === 'completed' ? 'bg-green-500' :
                        step.status === 'processing' ? 'bg-blue-500' :
                        'bg-red-500'
                      }`}
                      initial={{ width: 0 }}
                      animate={{ width: `${step.progress}%` }}
                      transition={{ duration: 0.3 }}
                    />
                  </div>
                  <p className="text-xs opacity-75">{step.message}</p>
                </>
              )}
            </motion.div>
          ))}
        </div>
      </div>

      {/* Latest Products Preview */}
      {processedProducts.length > 0 && (
        <div className="premium-card p-6">
          <h3 className="font-semibold mb-4">
            🎯 ผลลัพธ์ล่าสุด ({processedProducts.length} รายการ):
          </h3>
          <div className="space-y-2 max-h-60 overflow-y-auto">
            <AnimatePresence>
              {processedProducts.slice(-5).reverse().map((product, idx) => (
                <motion.div
                  key={product.id}
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="bg-white border border-gray-200 rounded-lg p-3 text-sm"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <p className="font-medium text-gray-900 mb-1">
                        {product.name_th}
                      </p>
                      <div className="flex items-center space-x-2 text-xs text-gray-600">
                        <span className="bg-blue-50 px-2 py-1 rounded">
                          {product.suggested_category.name_th}
                        </span>
                        <span className={`
                          px-2 py-1 rounded font-medium
                          ${product.suggested_category.confidence_score >= 0.7 
                            ? 'bg-green-50 text-green-700'
                            : product.suggested_category.confidence_score >= 0.4
                            ? 'bg-yellow-50 text-yellow-700'
                            : 'bg-red-50 text-red-700'
                          }
                        `}>
                          {(product.suggested_category.confidence_score * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 border border-red-200 p-4 rounded-lg">
          <div className="flex items-start space-x-3">
            <AlertCircleIcon className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
            <div>
              <h4 className="font-semibold text-red-800 mb-1">เกิดข้อผิดพลาด</h4>
              <p className="text-sm text-red-700">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex justify-between">
        {onBack && (
          <button
            onClick={onBack}
            disabled={isProcessing}
            className="btn-secondary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            ← ย้อนกลับ
          </button>
        )}
        <button
          onClick={() => onComplete(processedProducts)}
          disabled={isProcessing || processedProducts.length === 0}
          className="btn-premium disabled:opacity-50 disabled:cursor-not-allowed ml-auto"
        >
          {isProcessing
            ? 'กำลังประมวลผล...'
            : `ถัดไป: ตรวจสอบ (${processedProducts.length} รายการ) →`
          }
        </button>
      </div>
    </div>
  )
}
