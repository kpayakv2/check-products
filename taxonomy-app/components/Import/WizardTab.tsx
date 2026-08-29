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
import type { SaveResult, WizardItem } from '@/types/import'

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

/**
 * ไอดีประจำหนึ่งรอบการนำเข้า — ฝั่งเซิร์ฟเวอร์ใช้กันไม่ให้บันทึกซ้ำ
 *
 * crypto.randomUUID มีเฉพาะใน secure context เครื่อง LAN ที่เข้าผ่าน http://192.168.x.x
 * จะได้ undefined ซึ่งเป็นวิธีใช้งานจริงของระบบนี้ จึงต้องมีทางสำรองเสมอ
 */
const newRunId = () =>
  globalThis.crypto?.randomUUID?.() ??
  `${Date.now()}-${Math.random().toString(16).slice(2)}-${Math.random().toString(16).slice(2)}`

/** ชื่อที่ใช้อ้างอิงสินค้าหนึ่งรายการ ต้องเรียงลำดับให้ตรงกับตอนบันทึกใน commitDedup */
const itemName = (item: WizardItem): string => {
  for (const value of [item.name_th, item.name, item._cleaned_name]) {
    if (typeof value === 'string' && value) return value
  }
  return ''
}

export default function WizardTab() {
  const [currentStep, setCurrentStep] = useState(0)
  const [importMode, setImportMode] = useState<'upload' | 'storage'>('upload')
  const [file, setFile] = useState<File | null>(null)
  const [columnMapping, setColumnMapping] = useState<any | null>(null)
  
  // Pipeline State
  const [parsedData, setParsedData] = useState<ParsedCSV | null>(null)
  const [cleanedData, setCleanedData] = useState<WizardItem[]>([])
  const [dedupedData, setDedupedData] = useState<WizardItem[]>([])
  const [categorizedData, setCategorizedData] = useState<WizardItem[]>([])
  // ผลการบันทึกจริงจาก /api/import/commit — ขั้นสุดท้ายต้องแสดงตัวเลขจากตรงนี้
  // ไม่ใช่เดาจาก state ในเบราว์เซอร์ ไม่งั้นจะบอกว่าสำเร็จทั้งที่ยังไม่ได้บันทึก
  const [saveResult, setSaveResult] = useState<SaveResult | null>(null)
  const [saveError, setSaveError] = useState<string | null>(null)
  // ค่าเดิมตลอดหนึ่งรอบ wizard และเปลี่ยนใหม่เมื่อกดเริ่มใหม่เท่านั้น
  // ทำให้กดย้อน step แล้วเดินหน้าซ้ำก็ยังเป็นการบันทึกครั้งเดิม ไม่เกิด batch ใหม่
  const [runId, setRunId] = useState(newRunId)

  const handleNext = () => {
    if (currentStep < wizardSteps.length - 1) {
      setCurrentStep(currentStep + 1)
    }
  }

  /** บันทึกผลตรวจของซ้ำลงฐานข้อมูล แล้วจำ id ของแต่ละสินค้าไว้ให้ขั้นจัดหมวดใช้ต่อ */
  const commitDedup = async (deduped: WizardItem[]) => {
    // ห้ามบันทึกข้อมูลจำลองเด็ดขาด — fallback ของขั้นตรวจของซ้ำใช้คะแนนสุ่ม
    // ถ้า backend ล่มแล้วเผลอบันทึก จะได้สินค้าที่จัดกลุ่มมั่วเข้าฐานข้อมูลจริง
    if (deduped.some((item) => item._source === 'mock')) {
      setSaveError('ระบบวิเคราะห์ไม่พร้อมใช้งาน จึงยังไม่บันทึก — กรุณาลองใหม่เมื่อ backend กลับมา')
      return
    }

    setSaveError(null)
    try {
      const response = await fetch('/api/import/commit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'dedup',
          wizard_run_id: runId,
          file_name: file?.name,
          items: deduped.map((item) => ({
            name_th: itemName(item),
            cleaned_name: item._cleaned_name,
            bucket: item._bucket ?? 'new',
            similarity: item._similarity_score ?? 0,
            matched_product_id: item._matched_id,
            price: item.price,
          })),
        }),
      })
      const payload = await response.json()
      if (!payload.success) throw new Error(payload.error || 'บันทึกไม่สำเร็จ')
      setSaveResult(payload)
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : 'บันทึกไม่สำเร็จ')
    }
  }

  /** เติมหมวดหมู่ให้สินค้าที่บันทึกไว้แล้ว — จับคู่ด้วยชื่อเพราะ id อยู่ในผลลัพธ์ของขั้นก่อน */
  const commitCategories = async (categorized: WizardItem[]) => {
    // เดิมทั้งสองกรณีนี้ `return` เงียบ ๆ ผู้ใช้จึงเห็นหน้าจอ "บันทึกเข้าฐานข้อมูลแล้ว"
    // ทั้งที่หมวดหมู่ไม่เคยลงฐานข้อมูลเลย — ต้องรายงานออกไปเสมอ
    if (!saveResult?.import_batch_id) {
      setSaveError('ยังไม่ได้บันทึกรายการจากขั้นตรวจของซ้ำ จึงบันทึกหมวดหมู่ไม่ได้')
      return
    }

    const idByName = new Map<string, string>(
      (saveResult.products ?? []).map((p) => [p.name_th, p.id] as [string, string])
    )
    const assignments = categorized
      .map((item) => {
        // ชื่อที่ใช้บันทึกตอน commitDedup คือ _cleaned_name (ข้อมูลจาก CSV ไม่มี name_th)
        // คอลัมน์ดิบจาก CSV เป็น unknown จึงต้องกรองเอาเฉพาะที่เป็นข้อความก่อน
        const productId = idByName.get(itemName(item))
        // CategorizationStep เก็บผลไว้ในคีย์ที่ขึ้นต้นด้วย _ ไม่ใช่ object suggested_category
        const categoryId = item._suggested_category_id
        if (!productId || !categoryId) return null
        return {
          product_id: productId,
          category_id: categoryId,
          category_name: item._suggested_category,
          confidence_score: item._confidence ?? 0,
        }
      })
      .filter(Boolean)

    if (assignments.length === 0) {
      setSaveError('จับคู่สินค้ากับรายการที่บันทึกไว้ไม่ได้ หมวดหมู่จึงยังไม่ถูกบันทึก')
      return
    }

    try {
      const response = await fetch('/api/import/commit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: 'categorize',
          wizard_run_id: runId,
          import_batch_id: saveResult.import_batch_id,
          assignments,
        }),
      })
      const payload = await response.json()
      if (!payload.success) throw new Error(payload.error || 'บันทึกหมวดหมู่ไม่สำเร็จ')
      setSaveResult({ ...saveResult, categorized: payload.updated })
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : 'บันทึกหมวดหมู่ไม่สำเร็จ')
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
              onComplete={(cleaned) => {
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
              onComplete={(deduped) => {
                setDedupedData(deduped)
                // บันทึกตั้งแต่ตรงนี้ ไม่รอขั้นสุดท้าย — ถ้าปิดเบราว์เซอร์กลางคัน
                // รายการก้ำกึ่งที่ยังตรวจไม่จบต้องยังอยู่ให้ไปทำต่อที่หน้า Verify ได้
                commitDedup(deduped)
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
  
        case 3: {
          // เฉพาะของใหม่ (_bucket === 'new') เท่านั้นที่ต้องจัดหมวด — รายการซ้ำแน่นอน
          // ('duplicate') ถูก reject ไปแล้ว และรายการก้ำกึ่ง ('review') ยังไม่รู้ว่าจะกลาย
          // เป็นสินค้าจริงหรือถูกรวมเป็นตัวซ้ำ จึงจัดหมวดตอนนี้ไปก็อาจเสียเปล่า
          const newItems = dedupedData.filter((item) => item._bucket === 'new')
          return newItems.length > 0 ? (
            <CategorizationStep
              dedupedData={newItems}
              onComplete={(categorized) => {
                setCategorizedData(categorized)
                commitCategories(categorized)
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
        }

        case 4:
          return (
            <CompleteStep
              categorizedData={categorizedData}
              saveResult={saveResult}
              saveError={saveError}
              onReset={() => {
                setCurrentStep(0)
                setFile(null)
                setColumnMapping(null)
                setParsedData(null)
                setCleanedData([])
                setDedupedData([])
                setCategorizedData([])
                // ต้องล้างผลบันทึกด้วย ไม่งั้นรอบถัดไปที่บันทึกไม่สำเร็จจะยังชี้ batch เก่าอยู่
                // แล้วหมวดหมู่จะไปแปะสินค้าของรอบก่อน และหน้าสรุปจะโชว์ตัวเลขของรอบก่อน
                setSaveResult(null)
                setSaveError(null)
                setRunId(newRunId())
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
