'use client'

import { motion } from 'framer-motion'
import { CheckCircleIcon, DatabaseIcon, PlusIcon } from 'lucide-react'

export default function CompleteStep({
  categorizedData,
  onReset
}: any) {
  const approvedCount = categorizedData.filter((s: any) => s._status === 'approve').length
  const rejectedCount = categorizedData.filter((s: any) => s._status === 'reject').length
  const autoAssignedCount = categorizedData.length - approvedCount - rejectedCount // Fallback for pending

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
            className="w-32 h-32 bg-emerald-500 text-white rounded-full flex items-center justify-center mx-auto mb-8 shadow-2xl shadow-emerald-500/40"
          >
            <CheckCircleIcon className="w-16 h-16" />
          </motion.div>

          <h2 className="text-4xl font-black mb-4 font-noto-sans-thai text-slate-900">
            นำเข้าข้อมูลสำเร็จ!
          </h2>
          <p className="text-lg text-slate-500 mb-12 thai-text max-w-lg mx-auto">
            ระบบได้บันทึกข้อมูลสินค้าพร้อมหมวดหมู่ใหม่เข้าสู่ฐานข้อมูลเรียบร้อยแล้ว
          </p>

          <div className="grid grid-cols-3 gap-6 mb-12">
            <div className="bg-slate-50 p-6 rounded-3xl border border-slate-100">
              <p className="text-5xl font-black text-indigo-600 mb-2">{categorizedData.length}</p>
              <p className="text-sm font-bold text-slate-500 uppercase tracking-wider thai-text">ทั้งหมดที่ประมวลผล</p>
            </div>
            <div className="bg-emerald-50 p-6 rounded-3xl border border-emerald-100">
              <p className="text-5xl font-black text-emerald-600 mb-2">{approvedCount}</p>
              <p className="text-sm font-bold text-emerald-600 uppercase tracking-wider thai-text">อนุมัติแล้ว</p>
            </div>
            <div className="bg-rose-50 p-6 rounded-3xl border border-rose-100">
              <p className="text-5xl font-black text-rose-600 mb-2">{rejectedCount}</p>
              <p className="text-sm font-bold text-rose-600 uppercase tracking-wider thai-text">ปฏิเสธ / ยกเลิก</p>
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
