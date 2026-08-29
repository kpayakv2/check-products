'use client'

import { motion } from 'framer-motion'
import { AlertTriangleIcon, CheckCircleIcon, DatabaseIcon, PlusIcon } from 'lucide-react'
import type { SaveResult, WizardItem } from '@/types/import'

/**
 * สรุปผลการนำเข้า
 *
 * ตัวเลขทุกตัวต้องมาจาก `saveResult` ซึ่งเป็นผลตอบกลับจริงจาก /api/import/commit
 * ห้ามนับจาก state ในเบราว์เซอร์ — เดิมหน้านี้ขึ้นว่า "บันทึกเรียบร้อยแล้ว" ทุกครั้ง
 * ทั้งที่ทั้ง wizard ไม่เคยเขียนฐานข้อมูลเลยสักแถว
 */
interface CompleteStepProps {
  categorizedData: WizardItem[]
  saveResult: SaveResult | null
  saveError: string | null
  onReset: () => void
}

export default function CompleteStep({
  categorizedData,
  saveResult,
  saveError,
  onReset
}: CompleteStepProps) {
  const saved = saveResult?.saved ?? 0
  const counts = saveResult?.counts ?? {}
  const succeeded = !saveError && saved > 0
  const missingEmbedding = saveResult?.missing_embedding ?? 0

  return (
    <div className="max-w-4xl mx-auto space-y-8 py-10">
      <div className="premium-card p-12 text-center bg-white border border-slate-100 shadow-2xl rounded-[3rem] relative overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-emerald-500/10 rounded-full blur-[100px] -mr-48 -mt-48" />
        <div className="absolute bottom-0 left-0 w-96 h-96 bg-indigo-500/10 rounded-full blur-[100px] -ml-48 -mb-48" />

        <div className="relative z-10">
          <motion.div
            initial={{ scale: 0, rotate: -180 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{ type: "spring", duration: 0.8, bounce: 0.5 }}
            className={`w-32 h-32 text-white rounded-full flex items-center justify-center mx-auto mb-8 shadow-2xl ${
              succeeded ? 'bg-emerald-500 shadow-emerald-500/40' : 'bg-amber-500 shadow-amber-500/40'
            }`}
          >
            {succeeded ? <CheckCircleIcon className="w-16 h-16" /> : <AlertTriangleIcon className="w-16 h-16" />}
          </motion.div>

          <h2 className="text-4xl font-black mb-4 font-noto-sans-thai text-slate-900">
            {succeeded ? 'บันทึกเข้าฐานข้อมูลแล้ว' : 'ยังไม่ได้บันทึก'}
          </h2>
          <p className="text-lg text-slate-500 mb-12 thai-text max-w-xl mx-auto">
            {succeeded
              ? 'สินค้ารอตรวจอยู่ที่หน้า Data Quality → Verify ทำต่อจากที่ค้างไว้ได้'
              : saveError || 'ไม่มีข้อมูลถูกบันทึก — กรุณาตรวจสอบแล้วลองใหม่'}
          </p>

          {succeeded && missingEmbedding > 0 && (
            <div className="mb-8 p-4 bg-amber-50 border border-amber-200 rounded-2xl text-sm font-bold text-amber-800 thai-text">
              ⚠️ มี {missingEmbedding} รายการที่ไม่มี embedding — จะไม่ถูกนำไปเทียบในการตรวจของซ้ำครั้งหน้า
            </div>
          )}

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12">
            <div className="bg-slate-50 p-6 rounded-3xl border border-slate-100">
              <p className="text-4xl font-black text-slate-700 mb-2">{categorizedData.length}</p>
              <p className="text-xs font-bold text-slate-500 uppercase tracking-wider thai-text">ประมวลผล</p>
            </div>
            <div className="bg-indigo-50 p-6 rounded-3xl border border-indigo-100">
              <p className="text-4xl font-black text-indigo-600 mb-2">{counts.pending_review_category ?? 0}</p>
              <p className="text-xs font-bold text-indigo-600 uppercase tracking-wider thai-text">ของใหม่ รอจัดหมวด</p>
            </div>
            <div className="bg-amber-50 p-6 rounded-3xl border border-amber-100">
              <p className="text-4xl font-black text-amber-600 mb-2">{counts.pending_review_dedup ?? 0}</p>
              <p className="text-xs font-bold text-amber-600 uppercase tracking-wider thai-text">ก้ำกึ่ง รอตรวจ</p>
            </div>
            <div className="bg-rose-50 p-6 rounded-3xl border border-rose-100">
              <p className="text-4xl font-black text-rose-600 mb-2">{counts.rejected ?? 0}</p>
              <p className="text-xs font-bold text-rose-600 uppercase tracking-wider thai-text">มีในสตอกแล้ว</p>
            </div>
          </div>

          <div className="flex justify-center space-x-6">
            <button 
              onClick={onReset}
              className="px-8 py-4 bg-white border-2 border-slate-200 text-slate-700 hover:border-slate-300 hover:bg-slate-50 rounded-2xl font-bold transition-all flex items-center gap-2 thai-text shadow-sm"
            >
              <PlusIcon className="w-5 h-5" />
              นำเข้าไฟล์ใหม่
            </button>
            <button 
              onClick={() => window.location.href = '/products'}
              className="px-8 py-4 bg-slate-900 text-white hover:bg-black rounded-2xl font-bold transition-all flex items-center gap-2 thai-text shadow-xl shadow-slate-900/20 active:scale-95"
            >
              <DatabaseIcon className="w-5 h-5" />
              ไปที่คลังสินค้า
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
