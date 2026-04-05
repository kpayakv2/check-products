'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  AlertCircleIcon,
  CheckCircleIcon,
  InfoIcon,
  ArrowRightIcon
} from 'lucide-react'
import { parseCSV, validateCSV, getColumnStats, type ParsedCSV } from '@/utils/csv-parser'

export interface ColumnMapping {
  product_name: string      // Required
  description?: string      // Optional
  brand?: string           // Optional
  model?: string           // Optional
  price?: string           // Optional
  sku?: string            // Optional
  category?: string        // Optional (from similarity checker)
  confidence?: string      // Optional
  ignore: string[]         // Columns to ignore
}

interface ColumnMappingStepProps {
  file: File
  onComplete: (mapping: ColumnMapping, preview: ParsedCSV) => void
  onBack?: () => void
}

const FIELD_OPTIONS = [
  { value: '', label: '-- ไม่ใช้คอลัมน์นี้ --', required: false },
  { value: 'product_name', label: '🏷️ ชื่อสินค้า (จำเป็น)', required: true },
  { value: 'description', label: '📝 คำอธิบายสินค้า', required: false },
  { value: 'brand', label: '🏢 แบรนด์', required: false },
  { value: 'model', label: '📱 รุ่น/Model', required: false },
  { value: 'price', label: '💰 ราคา', required: false },
  { value: 'sku', label: '🔢 SKU/รหัสสินค้า', required: false },
  { value: 'category', label: '📂 หมวดหมู่เดิม', required: false },
  { value: 'confidence', label: '🎯 ความมั่นใจ', required: false }
]

export default function ColumnMappingStep({
  file,
  onComplete,
  onBack
}: ColumnMappingStepProps) {
  const [preview, setPreview] = useState<ParsedCSV | null>(null)
  const [columnMapping, setColumnMapping] = useState<Record<string, string>>({})
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [validation, setValidation] = useState<{
    isValid: boolean
    errors: string[]
    warnings: string[]
  } | null>(null)

  useEffect(() => {
    loadPreview()
  }, [file])

  const loadPreview = async () => {
    setIsLoading(true)
    setError(null)

    try {
      const text = await file.text()
      const parsed = parseCSV(text, { maxRows: 10 })
      const validation = validateCSV(parsed)

      setPreview(parsed)
      setValidation(validation)

      // Auto-detect column mapping
      const autoMapping: Record<string, string> = {}
      parsed.headers.forEach(header => {
        const lowerHeader = header.toLowerCase()
        
        if (lowerHeader.includes('product') || lowerHeader.includes('name') || lowerHeader === 'product_name') {
          autoMapping[header] = 'product_name'
        } else if (lowerHeader.includes('desc')) {
          autoMapping[header] = 'description'
        } else if (lowerHeader.includes('brand')) {
          autoMapping[header] = 'brand'
        } else if (lowerHeader.includes('model')) {
          autoMapping[header] = 'model'
        } else if (lowerHeader.includes('price') || lowerHeader.includes('ราคา')) {
          autoMapping[header] = 'price'
        } else if (lowerHeader.includes('sku') || lowerHeader.includes('รหัส')) {
          autoMapping[header] = 'sku'
        } else if (lowerHeader.includes('category') || lowerHeader.includes('หมวด')) {
          autoMapping[header] = 'category'
        } else if (lowerHeader.includes('confidence')) {
          autoMapping[header] = 'confidence'
        }
      })

      setColumnMapping(autoMapping)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'เกิดข้อผิดพลาดในการอ่านไฟล์')
    } finally {
      setIsLoading(false)
    }
  }

  const handleMappingChange = (header: string, value: string) => {
    setColumnMapping(prev => {
      const newMapping = { ...prev }
      
      // Remove if selecting empty
      if (value === '') {
        delete newMapping[header]
      } else {
        // Remove old mapping for this value
        Object.keys(newMapping).forEach(key => {
          if (newMapping[key] === value && key !== header) {
            delete newMapping[key]
          }
        })
        newMapping[header] = value
      }
      
      return newMapping
    })
  }

  const handleComplete = () => {
    if (!preview) return

    const mapping: ColumnMapping = {
      product_name: '',
      ignore: []
    }

    // Build mapping
    Object.entries(columnMapping).forEach(([header, field]) => {
      if (field === 'product_name') mapping.product_name = header
      else if (field === 'description') mapping.description = header
      else if (field === 'brand') mapping.brand = header
      else if (field === 'model') mapping.model = header
      else if (field === 'price') mapping.price = header
      else if (field === 'sku') mapping.sku = header
      else if (field === 'category') mapping.category = header
      else if (field === 'confidence') mapping.confidence = header
    })

    // Add ignored columns
    preview.headers.forEach(header => {
      if (!columnMapping[header]) {
        mapping.ignore.push(header)
      }
    })

    onComplete(mapping, preview)
  }

  const isProductNameMapped = Object.values(columnMapping).includes('product_name')
  const canProceed = isProductNameMapped && validation?.isValid

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">กำลังอ่านไฟล์...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="premium-card p-8">
        <div className="flex items-start space-x-4 text-red-600">
          <AlertCircleIcon className="w-6 h-6 flex-shrink-0 mt-1" />
          <div>
            <h3 className="font-semibold mb-2">เกิดข้อผิดพลาด</h3>
            <p className="text-sm">{error}</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="premium-card p-6">
        <h2 className="text-2xl font-bold mb-2 font-noto-sans-thai">
          📊 กำหนดการจับคู่คอลัมน์
        </h2>
        <p className="text-gray-600">
          เลือกว่าคอลัมน์ไหนในไฟล์คือชื่อสินค้า (จำเป็น) และข้อมูลอื่นๆ
        </p>
      </div>

      {/* Validation Messages */}
      {validation && (validation.errors.length > 0 || validation.warnings.length > 0) && (
        <div className="space-y-2">
          {validation.errors.map((error, idx) => (
            <div key={idx} className="bg-red-50 border border-red-200 p-4 rounded-lg flex items-start space-x-3">
              <AlertCircleIcon className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
              <p className="text-sm text-red-800">{error}</p>
            </div>
          ))}
          {validation.warnings.map((warning, idx) => (
            <div key={idx} className="bg-yellow-50 border border-yellow-200 p-4 rounded-lg flex items-start space-x-3">
              <InfoIcon className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
              <p className="text-sm text-yellow-800">{warning}</p>
            </div>
          ))}
        </div>
      )}

      {/* Preview Table */}
      {preview && (
        <div className="premium-card p-6">
          <h3 className="text-lg font-semibold mb-4 font-noto-sans-thai">
            ตัวอย่างข้อมูล ({preview.rows.length} แถวแรกจาก {preview.totalCount} แถวทั้งหมด)
          </h3>

          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  {preview.headers.map((header, idx) => (
                    <th
                      key={idx}
                      className="px-4 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider"
                    >
                      <div className="space-y-2">
                        <div className="font-semibold">{header}</div>
                        
                        {/* Column Selector */}
                        <select
                          value={columnMapping[header] || ''}
                          onChange={(e) => handleMappingChange(header, e.target.value)}
                          className={`
                            w-full text-sm rounded border px-2 py-1.5
                            ${columnMapping[header] === 'product_name' 
                              ? 'border-blue-500 bg-blue-50 text-blue-900 font-semibold' 
                              : columnMapping[header]
                              ? 'border-green-500 bg-green-50 text-green-900'
                              : 'border-gray-300 bg-white text-gray-700'
                            }
                          `}
                        >
                          {FIELD_OPTIONS.map(option => (
                            <option key={option.value} value={option.value}>
                              {option.label}
                            </option>
                          ))}
                        </select>

                        {/* Column Stats */}
                        {preview && (() => {
                          const stats = getColumnStats(preview, header)
                          return (
                            <div className="text-xs text-gray-500 space-y-0.5">
                              <div>ค่าที่ไม่ว่าง: {stats.totalValues - stats.emptyValues}/{stats.totalValues}</div>
                              <div>ค่าไม่ซ้ำ: {stats.uniqueValues}</div>
                            </div>
                          )
                        })()}
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {preview.rows.map((row, rowIdx) => (
                  <tr key={rowIdx} className={rowIdx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    {preview.headers.map((header, colIdx) => (
                      <td key={colIdx} className="px-4 py-3 text-sm text-gray-900 whitespace-nowrap">
                        <div className="max-w-xs truncate" title={row[header]}>
                          {row[header] || <span className="text-gray-400">-</span>}
                        </div>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Mapping Summary */}
      <div className="premium-card p-6">
        <h3 className="text-lg font-semibold mb-4 font-noto-sans-thai">
          สรุปการจับคู่คอลัมน์
        </h3>

        <div className="space-y-3">
          <div className={`flex items-center space-x-3 ${isProductNameMapped ? 'text-green-600' : 'text-red-600'}`}>
            {isProductNameMapped ? (
              <CheckCircleIcon className="w-5 h-5" />
            ) : (
              <AlertCircleIcon className="w-5 h-5" />
            )}
            <span className="font-medium">
              {isProductNameMapped 
                ? '✅ ได้เลือกคอลัมน์ชื่อสินค้าแล้ว' 
                : '❌ ยังไม่ได้เลือกคอลัมน์ชื่อสินค้า (จำเป็น)'}
            </span>
          </div>

          {preview && (
            <div className="bg-gray-50 p-4 rounded-lg space-y-2 text-sm">
              <div className="font-semibold">คอลัมน์ที่เลือก:</div>
              <ul className="space-y-1 ml-4">
                {Object.entries(columnMapping).map(([header, field]) => {
                  const option = FIELD_OPTIONS.find(o => o.value === field)
                  return (
                    <li key={header} className="flex items-center space-x-2">
                      <ArrowRightIcon className="w-4 h-4 text-gray-400" />
                      <span className="font-mono bg-white px-2 py-0.5 rounded">{header}</span>
                      <span className="text-gray-600">→</span>
                      <span className="text-blue-600">{option?.label}</span>
                    </li>
                  )
                })}
              </ul>

              {preview.headers.filter(h => !columnMapping[h]).length > 0 && (
                <>
                  <div className="font-semibold mt-3">คอลัมน์ที่ไม่ใช้:</div>
                  <div className="flex flex-wrap gap-2 ml-4">
                    {preview.headers
                      .filter(h => !columnMapping[h])
                      .map(h => (
                        <span key={h} className="font-mono bg-gray-200 text-gray-600 px-2 py-0.5 rounded text-xs">
                          {h}
                        </span>
                      ))
                    }
                  </div>
                </>
              )}

              <div className="mt-3 pt-3 border-t border-gray-200">
                <div>จำนวนสินค้าที่จะประมวลผล: <strong>{preview.totalCount}</strong> รายการ</div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex justify-between">
        {onBack && (
          <button onClick={onBack} className="btn-secondary">
            ← ย้อนกลับ
          </button>
        )}
        <button
          onClick={handleComplete}
          disabled={!canProceed}
          className="btn-premium disabled:opacity-50 disabled:cursor-not-allowed ml-auto"
        >
          ถัดไป: เริ่มประมวลผล AI →
        </button>
      </div>
    </div>
  )
}
