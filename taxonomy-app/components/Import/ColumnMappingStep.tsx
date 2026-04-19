'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  AlertCircleIcon,
  CheckCircleIcon,
  InfoIcon,
  ArrowRightIcon,
  TagIcon,
  FileTextIcon,
  BriefcaseIcon,
  SmartphoneIcon,
  CircleDollarSignIcon,
  FingerprintIcon,
  FolderTreeIcon,
  TargetIcon,
  ListFilterIcon,
  EyeIcon
} from 'lucide-react'
import { parseCSV, validateCSV, getColumnStats, type ParsedCSV } from '@/utils/csv-parser'

export interface ColumnMapping {
  product_name: string
  product_name_index?: number
  description?: string
  description_index?: number
  brand?: string
  brand_index?: number
  model?: string
  model_index?: number
  price?: string
  price_index?: number
  sku?: string
  sku_index?: number
  category?: string
  category_index?: number
  confidence?: string
  confidence_index?: number
  ignore: string[]
}

interface ColumnMappingStepProps {
  file: File
  onComplete: (mapping: ColumnMapping, preview: ParsedCSV) => void
  onBack?: () => void
}

interface FieldDefinition {
  key: keyof Omit<ColumnMapping, 'ignore'>
  label: string
  icon: any
  required: boolean
  description: string
  thaiKeywords: string[]
}

const SYSTEM_FIELDS: FieldDefinition[] = [
  { 
    key: 'product_name', 
    label: 'ชื่อสินค้า', 
    icon: TagIcon, 
    required: true, 
    description: 'คอลัมน์ที่เก็บชื่อสินค้าภาษาไทย (สำคัญที่สุด)',
    thaiKeywords: ['รายการ', 'สินค้า', 'ชื่อ', 'product', 'name']
  },
  { 
    key: 'price', 
    label: 'ราคาขาย', 
    icon: CircleDollarSignIcon, 
    required: false, 
    description: 'ราคาสินค้าต่อหน่วย',
    thaiKeywords: ['ราคา', 'price', 'เงิน', 'บาท']
  },
  { 
    key: 'sku', 
    label: 'รหัสสินค้า/Barcode', 
    icon: FingerprintIcon, 
    required: false, 
    description: 'รหัส SKU, บาร์โค้ด หรือรหัสอ้างอิงอื่นๆ',
    thaiKeywords: ['รหัส', 'barcode', 'sku', 'id']
  },
  { 
    key: 'description', 
    label: 'รายละเอียดสินค้า', 
    icon: FileTextIcon, 
    required: false, 
    description: 'ข้อมูลเพิ่มเติมเกี่ยวกับสินค้า',
    thaiKeywords: ['รายละเอียด', 'desc']
  },
  { 
    key: 'brand', 
    label: 'แบรนด์/ยี่ห้อ', 
    icon: BriefcaseIcon, 
    required: false, 
    description: 'ชื่อยี่ห้อสินค้า',
    thaiKeywords: ['แบรนด์', 'ยี่ห้อ', 'brand']
  },
  { 
    key: 'model', 
    label: 'รุ่น (Model)', 
    icon: SmartphoneIcon, 
    required: false, 
    description: 'รุ่นของสินค้า',
    thaiKeywords: ['รุ่น', 'model']
  },
  { 
    key: 'category', 
    label: 'หมวดหมู่เดิม', 
    icon: FolderTreeIcon, 
    required: false, 
    description: 'หมวดหมู่ที่มีอยู่แล้วในไฟล์',
    thaiKeywords: ['หมวด', 'cat']
  }
]

export default function ColumnMappingStep({
  file,
  onComplete,
  onBack
}: ColumnMappingStepProps) {
  const [preview, setPreview] = useState<ParsedCSV | null>(null)
  // New state structure: system_field -> csv_header
  const [mapping, setMapping] = useState<Record<string, string>>({})
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

      // Auto-detect mapping
      const autoMapping: Record<string, string> = {}
      
      SYSTEM_FIELDS.forEach(field => {
        const foundHeader = parsed.headers.find(header => {
          const lower = header.toLowerCase()
          return field.thaiKeywords.some(kw => lower.includes(kw.toLowerCase()))
        })
        
        if (foundHeader) {
          autoMapping[field.key] = foundHeader
        }
      })

      setMapping(autoMapping)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'เกิดข้อผิดพลาดในการอ่านไฟล์')
    } finally {
      setIsLoading(false)
    }
  }

  const handleFieldChange = (fieldKey: string, headerName: string) => {
    setMapping(prev => {
      const newMapping = { ...prev }
      if (headerName === '') {
        delete newMapping[fieldKey]
      } else {
        newMapping[fieldKey] = headerName
      }
      return newMapping
    })
  }

  const handleComplete = () => {
    if (!preview) return

    const finalMapping: ColumnMapping = {
      product_name: mapping.product_name || '',
      ignore: []
    }

    // Map system fields to ColumnMapping object
    SYSTEM_FIELDS.forEach(field => {
      const header = mapping[field.key]
      if (header) {
        const idx = preview.headers.indexOf(header)
        if (idx !== -1) {
          (finalMapping as any)[field.key] = header;
          (finalMapping as any)[`${field.key}_index`] = idx;
        }
      }
    })

    // Find ignored columns
    const mappedHeaders = new Set(Object.values(mapping))
    preview.headers.forEach(header => {
      if (!mappedHeaders.has(header)) {
        finalMapping.ignore.push(header)
      }
    })

    onComplete(finalMapping, preview)
  }

  const isProductNameMapped = !!mapping.product_name
  const canProceed = isProductNameMapped && validation?.isValid

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center py-20 space-y-4">
        <div className="relative">
          <div className="w-16 h-16 border-4 border-indigo-100 border-t-indigo-600 rounded-full animate-spin"></div>
          <BrainIcon className="w-6 h-6 text-indigo-600 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
        </div>
        <p className="text-slate-500 font-bold animate-pulse font-noto-sans-thai text-sm uppercase tracking-widest">พยัคฆ์ AI กำลังตรวจสอบโครงสร้างไฟล์...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="premium-card p-12 bg-rose-50 border-rose-100 flex flex-col items-center text-center">
        <div className="w-16 h-16 bg-rose-100 text-rose-600 rounded-3xl flex items-center justify-center mb-6">
          <AlertCircleIcon className="w-8 h-8" />
        </div>
        <h3 className="text-xl font-black text-rose-900 mb-2">อ๊ะ! เกิดข้อผิดพลาด</h3>
        <p className="text-rose-600 font-medium mb-8 max-w-md">{error}</p>
        <button onClick={onBack} className="btn-secondary">กลับไปอัปโหลดใหม่</button>
      </div>
    )
  }

  return (
    <div className="max-w-5xl mx-auto space-y-8 pb-20">
      {/* Header Info */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-black text-slate-900 tracking-tight font-noto-sans-thai mb-2">
            📊 จับคู่คอลัมน์ (Column Mapping)
          </h2>
          <p className="text-slate-500 font-medium">ระบุว่าข้อมูลแต่ละหัวข้อระบบ อยู่ที่คอลัมน์ไหนในไฟล์ของพี่กานครับ</p>
        </div>
        <div className="bg-indigo-50 px-4 py-2 rounded-2xl flex items-center gap-3">
          <div className="w-8 h-8 bg-indigo-600 text-white rounded-xl flex items-center justify-center font-black text-xs">
            {preview?.totalCount}
          </div>
          <span className="text-xs font-black text-indigo-900 uppercase tracking-widest">รายการทั้งหมด</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: The Form */}
        <div className="lg:col-span-7 space-y-6">
          <div className="premium-card p-8 bg-white border border-slate-100 shadow-xl shadow-slate-200/50">
            <div className="flex items-center gap-3 mb-8 pb-4 border-b border-slate-50">
              <ListFilterIcon className="w-5 h-5 text-indigo-600" />
              <h3 className="text-sm font-black text-slate-900 uppercase tracking-widest">ตั้งค่าคอลัมน์</h3>
            </div>

            <div className="space-y-6">
              {SYSTEM_FIELDS.map((field) => (
                <div key={field.key} className="group">
                  <div className="flex items-center justify-between mb-2">
                    <label className="flex items-center gap-2 text-sm font-black text-slate-700">
                      <field.icon className={`w-4 h-4 ${field.required ? 'text-indigo-600' : 'text-slate-400'}`} />
                      {field.label}
                      {field.required && <span className="text-rose-500">*</span>}
                    </label>
                    <span className="text-[10px] text-slate-400 font-medium opacity-0 group-hover:opacity-100 transition-opacity">
                      {field.description}
                    </span>
                  </div>

                  <div className="relative">
                    <select
                      value={mapping[field.key] || ''}
                      onChange={(e) => handleFieldChange(field.key, e.target.value)}
                      className={`
                        w-full appearance-none bg-slate-50 border-2 rounded-2xl px-5 py-3.5 pr-12
                        text-sm font-bold transition-all outline-none
                        ${mapping[field.key] 
                          ? 'border-indigo-100 text-slate-900 focus:border-indigo-500 bg-white' 
                          : field.required 
                          ? 'border-slate-100 text-slate-300 focus:border-indigo-500' 
                          : 'border-slate-50 text-slate-300'
                        }
                      `}
                    >
                      <option value="">-- เลือกคอลัมน์จากไฟล์ --</option>
                      {preview?.headers.map((header, idx) => {
                        const stats = getColumnStats(preview, header)
                        return (
                          <option key={header} value={header}>
                            {idx + 1}. {header} (เช่น: {stats.sampleValues.join(', ')})
                          </option>
                        )
                      })}
                    </select>
                    <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-slate-400">
                      <ArrowRightIcon className="w-4 h-4 rotate-90" />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column: Preview & Validation */}
        <div className="lg:col-span-5 space-y-6">
          {/* Validation Box */}
          <div className={`p-8 rounded-[2rem] border-2 transition-all ${
            isProductNameMapped ? 'bg-emerald-50/50 border-emerald-100' : 'bg-rose-50/50 border-rose-100'
          }`}>
            <div className="flex items-start gap-4">
              <div className={`w-12 h-12 rounded-2xl flex items-center justify-center shrink-0 ${
                isProductNameMapped ? 'bg-emerald-100 text-emerald-600' : 'bg-rose-100 text-rose-600'
              }`}>
                {isProductNameMapped ? <CheckCircleIcon className="w-6 h-6" /> : <AlertCircleIcon className="w-6 h-6" />}
              </div>
              <div>
                <h4 className={`text-lg font-black uppercase tracking-tight mb-1 ${
                  isProductNameMapped ? 'text-emerald-900' : 'text-rose-900'
                }`}>
                  {isProductNameMapped ? 'พร้อมประมวลผล' : 'ต้องการข้อมูลเพิ่ม'}
                </h4>
                <p className={`text-xs font-medium ${isProductNameMapped ? 'text-emerald-600' : 'text-rose-600'}`}>
                  {isProductNameMapped 
                    ? 'ระบบตรวจสอบเบื้องต้นแล้ว ข้อมูลครบถ้วนสำหรับเริ่มงานครับ' 
                    : 'กรุณาเลือกคอลัมน์ "ชื่อสินค้า" เพื่อให้ AI ทำงานได้'}
                </p>
              </div>
            </div>
          </div>

          {/* Data Sample Preview */}
          <div className="premium-card p-8 bg-white border border-slate-100 shadow-sm overflow-hidden">
             <div className="flex items-center gap-3 mb-6">
              <EyeIcon className="w-5 h-5 text-slate-400" />
              <h3 className="text-sm font-black text-slate-900 uppercase tracking-widest">ข้อมูลต้นฉบับ (5 แถวแรก)</h3>
            </div>
            
            <div className="overflow-x-auto -mx-8">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-slate-50">
                    {preview?.headers.map((h, i) => (
                      <th key={i} className="px-6 py-3 text-[10px] font-black text-slate-400 uppercase tracking-tighter border-b border-slate-100 whitespace-nowrap">
                        {i+1}. {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview?.rows.slice(0, 5).map((row, ri) => (
                    <tr key={ri} className="border-b border-slate-50 last:border-0">
                      {preview.headers.map((h, ci) => (
                        <td key={ci} className="px-6 py-4 text-xs font-bold text-slate-600 whitespace-nowrap max-w-[150px] truncate">
                          {row[h] || '-'}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-6 p-4 bg-slate-50 rounded-2xl text-[10px] font-bold text-slate-400 text-center uppercase tracking-widest">
              แสดงเฉพาะ 5 แถวแรกเพื่อใช้ตรวจสอบความถูกต้อง
            </div>
          </div>
        </div>
      </div>

      {/* Footer Actions */}
      <div className="flex items-center justify-between pt-8 border-t border-slate-100">
        <button
          onClick={onBack}
          className="px-8 py-4 text-slate-400 hover:text-slate-900 font-bold text-sm transition-all flex items-center gap-2"
        >
          ← ย้อนกลับ
        </button>

        <button
          onClick={handleComplete}
          disabled={!canProceed}
          className={`
            px-16 py-5 rounded-[2rem] font-black text-sm uppercase tracking-[0.2em] shadow-2xl transition-all active:scale-95 flex items-center gap-4
            ${canProceed 
              ? 'bg-slate-900 text-white hover:bg-black shadow-slate-900/30' 
              : 'bg-slate-100 text-slate-300 cursor-not-allowed shadow-none'
            }
          `}
        >
          {canProceed ? (
            <>
              เริ่มประมวลผล AI
              <ZapIcon className="w-4 h-4 fill-current text-yellow-400" />
            </>
          ) : (
            'รอเลือกชื่อสินค้า...'
          )}
        </button>
      </div>
    </div>
  )
}

function BrainIcon(props: any) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.54Z" />
      <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.54Z" />
    </svg>
  )
}
