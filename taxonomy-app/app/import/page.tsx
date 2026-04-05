'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import Link from 'next/link'
import Sidebar from '@/components/Layout/Sidebar'
import Header from '@/components/Layout/Header'
import { 
  UploadIcon,
  ClockIcon,
  CheckCircleIcon,
  ArrowRightIcon,
  FolderIcon,
  FileSpreadsheetIcon,
  ChevronRightIcon,
  ZapIcon,
  ShieldCheckIcon,
  DatabaseIcon,
  InboxIcon
} from 'lucide-react'

export default function ImportPage() {
  const [pendingCount, setPendingCount] = useState<number | null>(null)

  // Load pending count
  useEffect(() => {
    const loadPendingCount = async () => {
      try {
        const response = await fetch('/api/import/pending?limit=1')
        if (response.ok) {
          const data = await response.json()
          setPendingCount(data.pagination.total)
        }
      } catch (error) {
        console.error('Error loading pending count:', error)
      }
    }

    loadPendingCount()
  }, [])

  return (
    <div className="flex h-screen bg-gray-50 font-sans">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        
        <main className="flex-1 overflow-x-hidden overflow-y-auto p-8 relative">
          {/* Decorative Background Elements */}
          <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-indigo-50/50 rounded-full blur-[120px] -mr-48 -mt-48 pointer-events-none" />
          <div className="absolute bottom-10 left-10 w-[300px] h-[300px] bg-emerald-50/50 rounded-full blur-[100px] pointer-events-none" />

          <div className="max-w-5xl mx-auto relative z-10">
            {/* Page Header Area */}
            <div className="mb-12">
              <h1 className="text-4xl font-extrabold text-slate-900 tracking-tight thai-text">Import ข้อมูลสินค้า</h1>
              <p className="text-slate-500 mt-2 text-lg font-medium thai-text">นำเข้า ตรวจสอบ และอนุมัติข้อมูลสินค้าขนาดใหญ่อย่างอัจฉริยะ</p>
            </div>

            {/* Main Action Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
              {/* New Import Card */}
              <motion.div
                whileHover={{ y: -8 }}
                transition={{ type: "spring", stiffness: 400, damping: 20 }}
              >
                <Link href="/import/wizard" className="block group">
                  <div className="bg-white/60 backdrop-blur-md rounded-[48px] p-10 border border-white shadow-xl shadow-indigo-100/20 group-hover:bg-white group-hover:border-indigo-100 transition-all cursor-pointer h-full flex flex-col items-start">
                    <div className="w-16 h-16 bg-indigo-600 rounded-3xl flex items-center justify-center text-white shadow-lg shadow-indigo-600/30 mb-8 transform group-hover:scale-110 group-hover:rotate-3 transition-transform duration-300">
                      <UploadIcon className="w-8 h-8" />
                    </div>
                    
                    <h3 className="text-2xl font-black text-slate-900 mb-4 thai-text tracking-tight uppercase">
                      New Import Batch
                    </h3>
                    <p className="text-slate-500 text-base font-medium thai-text leading-relaxed mb-10 flex-1">
                      อัปโหลดผลิตภัณฑ์จากไฟล์ CSV ของคุณ ระบบจะวิเคราะห์โครงสร้างข้อมูลและแนะนำหมวดหมู่ตาม Taxonomy อัตโนมัติ
                    </p>
                    
                    <div className="flex items-center gap-2 text-indigo-600 font-black tracking-widest text-xs uppercase">
                      Start Migration Process
                      <ArrowRightIcon className="w-4 h-4 group-hover:translate-x-2 transition-transform" />
                    </div>
                  </div>
                </Link>
              </motion.div>

              {/* Pending Approvals Card */}
              <motion.div
                whileHover={{ y: -8 }}
                transition={{ type: "spring", stiffness: 400, damping: 20 }}
              >
                <Link href="/import/pending" className="block group">
                  <div className="bg-white/60 backdrop-blur-md rounded-[48px] p-10 border border-white shadow-xl shadow-amber-100/20 group-hover:bg-white group-hover:border-amber-100 transition-all cursor-pointer h-full flex flex-col items-start">
                    <div className="w-16 h-16 bg-amber-100 rounded-3xl flex items-center justify-center text-amber-600 border border-amber-200/50 mb-8 transform group-hover:scale-110 group-hover:-rotate-3 transition-transform duration-300">
                      <InboxIcon className="w-8 h-8" />
                    </div>
                    
                    <div className="flex items-center justify-between w-full mb-4">
                      <h3 className="text-2xl font-black text-slate-900 thai-text tracking-tight uppercase">
                        Pending Reviews
                      </h3>
                      {pendingCount !== null && (
                        <div className="bg-amber-600 text-white px-3 py-1.5 rounded-2xl text-xs font-black tracking-widest shadow-lg shadow-amber-600/20">
                          {pendingCount}
                        </div>
                      )}
                    </div>
                    
                    <p className="text-slate-500 text-base font-medium thai-text leading-relaxed mb-10 flex-1">
                      ตรวจสอบสินค้าที่ AI ประมวลผลเสร็จสิ้นแล้ว คุณสามารถปรับแก้ชื่อ คุณลักษณะ หรือสถานะได้รายชิ้นหรือแบบกลุ่ม
                    </p>
                    
                    <div className="flex items-center gap-2 text-amber-600 font-black tracking-widest text-xs uppercase">
                      Review Work Queue
                      <ArrowRightIcon className="w-4 h-4 group-hover:translate-x-2 transition-transform" />
                    </div>
                  </div>
                </Link>
              </motion.div>
            </div>

            {/* Advanced Utility Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16">
              <div className="bg-white p-8 rounded-[40px] border border-slate-100 flex items-center gap-6 group hover:border-indigo-100 transition-colors">
                <div className="w-14 h-14 bg-slate-50 rounded-2xl flex items-center justify-center text-slate-400 group-hover:bg-indigo-50 group-hover:text-indigo-600 transition-colors">
                  <FolderIcon className="w-6 h-6" />
                </div>
                <div>
                   <h4 className="text-sm font-black text-slate-900 uppercase tracking-widest mb-1">Local Storage</h4>
                   <p className="text-xs font-medium text-slate-400 thai-text leading-tight">จัดการไฟล์ที่เคยอัปโหลดไว้แล้ว</p>
                </div>
              </div>

              <div className="bg-white p-8 rounded-[40px] border border-slate-100 flex items-center gap-6 group hover:border-indigo-100 transition-colors">
                <div className="w-14 h-14 bg-slate-50 rounded-2xl flex items-center justify-center text-slate-400 group-hover:bg-indigo-50 group-hover:text-indigo-600 transition-colors">
                  <FileSpreadsheetIcon className="w-6 h-6" />
                </div>
                <div>
                   <h4 className="text-sm font-black text-slate-900 uppercase tracking-widest mb-1">CSV Template</h4>
                   <p className="text-xs font-medium text-slate-400 thai-text leading-relaxed">ดาวน์โหลดเทมเพลตมาตรฐาน</p>
                </div>
              </div>

              <div className="bg-white p-8 rounded-[40px] border border-slate-100 flex items-center gap-6 group hover:border-indigo-100 transition-colors">
                <div className="w-14 h-14 bg-slate-50 rounded-2xl flex items-center justify-center text-slate-400 group-hover:bg-indigo-50 group-hover:text-indigo-600 transition-colors">
                  <ZapIcon className="w-6 h-6" />
                </div>
                <div>
                   <h4 className="text-sm font-black text-slate-900 uppercase tracking-widest mb-1">Fast Track</h4>
                   <p className="text-xs font-medium text-slate-400 thai-text leading-relaxed">อัปโหลดแบบข้ามขั้นตอนรีวิว</p>
                </div>
              </div>
            </div>

            {/* Visual Workflow Timeline */}
            <div className="bg-slate-900 rounded-[56px] p-12 overflow-hidden relative shadow-2xl">
              <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-500/10 rounded-full blur-[100px] -mr-32 -mt-32" />
              
              <div className="relative mb-12 flex items-center justify-between">
                <div>
                  <h3 className="text-2xl font-black text-white thai-text uppercase tracking-tight">System Workflow</h3>
                  <p className="text-indigo-300 font-medium thai-text mt-1 leading-relaxed">อธิบายกระบวนการเตรียมข้อมูลและนำเข้าสินค้าอัตโนมัติ</p>
                </div>
                <div className="px-4 py-2 bg-indigo-500/20 text-indigo-200 border border-indigo-500/30 rounded-2xl text-xs font-black uppercase tracking-widest">
                  Standard Operation Procedure
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-4 gap-12 relative">
                {/* Horizontal line for desktop */}
                <div className="hidden lg:block absolute top-[28px] left-8 right-8 h-px bg-white/10" />

                {[
                  { icon: UploadIcon, step: 1, title: 'Upload & Parse', color: 'indigo', desc: 'ไฟล์ CSV จะถูกถอดโครงสร้างและตรวจสอบความถูกต้องเบื้องต้น' },
                  { icon: ZapIcon, step: 2, title: 'AI Classification', color: 'indigo', desc: 'ระบบวิเคราะห์ชื่อเพื่อจัดหมวดหมู่สินค้าตาม Taxonomy (Ph. 2)' },
                  { icon: ShieldCheckIcon, step: 3, title: 'Human Audit', color: 'indigo', desc: 'ทีมตรวจสอบยืนยันผลลัพธ์จาก AI เพื่อความแม่นยำสูงสุด' },
                  { icon: DatabaseIcon, step: 4, title: 'Final Sync', color: 'emerald', desc: 'ข้อมูลที่อนุมัติจะถูกจัดเก็บเข้าสู่ Inventory Database หลัก' }
                ].map((item, idx) => (
                  <div key={idx} className="relative flex lg:flex-col gap-6 lg:gap-8 items-start lg:items-center">
                    <div className={`w-14 h-14 rounded-2xl flex items-center justify-center bg-${item.color}-500 text-white shadow-xl shadow-${item.color}-500/20 z-10 border border-white/20 transform group-hover:scale-110 transition-transform`}>
                      <item.icon className="w-6 h-6" />
                    </div>
                    <div className="flex-1 lg:text-center">
                      <div className="text-xs font-black text-indigo-400 uppercase tracking-[0.2em] mb-2 font-mono">Step 0{item.step}</div>
                      <h4 className="text-lg font-black text-white uppercase tracking-tight mb-3 thai-text">{item.title}</h4>
                      <p className="text-slate-400 text-sm font-medium leading-relaxed thai-text">{item.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
