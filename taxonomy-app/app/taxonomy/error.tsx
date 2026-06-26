'use client'

import { useEffect } from 'react'
import { motion } from 'framer-motion'
import { FolderTreeIcon, RefreshCcwIcon, ArrowLeftIcon } from 'lucide-react'
import Link from 'next/link'

export default function TaxonomyError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  useEffect(() => {
    console.error('Taxonomy Page Error:', error)
  }, [error])

  return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center p-6 relative overflow-hidden">
      <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-indigo-50/50 rounded-full blur-[100px] -mr-48 -mt-48 pointer-events-none" />
      
      <motion.div 
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className="max-w-lg w-full bg-white/70 backdrop-blur-xl rounded-[48px] shadow-2xl p-12 border border-white relative z-10"
      >
        <div className="flex items-center gap-6 mb-10">
          <div className="w-16 h-16 bg-indigo-600 rounded-3xl flex items-center justify-center text-white shadow-lg shadow-indigo-100">
             <FolderTreeIcon className="w-8 h-8" />
          </div>
          <div>
            <h2 className="text-2xl font-black text-slate-900 thai-text uppercase tracking-tight">Taxonomy Error</h2>
            <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mt-1">Hierarchical Sync Failed</p>
          </div>
        </div>

        <div className="bg-rose-50/50 border border-rose-100/50 rounded-3xl p-6 mb-10">
           <p className="text-rose-600 text-sm thai-text leading-relaxed italic">
             "เกิดความล่าช้าหรือข้อผิดพลาดในการโหลดโครงสร้าง Taxonomy Tree กรุณาตรวจสอบการเชื่อมต่ออินเทอร์เน็ตหรือลองรีเฟรชหน้าจอนี้อีกครั้ง"
           </p>
        </div>

        <div className="flex gap-4">
          <button
            onClick={() => reset()}
            className="flex-1 flex items-center justify-center gap-3 bg-indigo-600 text-white py-5 rounded-3xl font-black uppercase tracking-widest hover:bg-indigo-700 transition-all shadow-xl shadow-indigo-200"
          >
            <RefreshCcwIcon className="w-5 h-5" />
            <span className="thai-text">Reload Tree</span>
          </button>
          
          <Link 
            href="/"
            className="flex-1 flex items-center justify-center gap-3 bg-white border border-slate-100 text-slate-400 py-5 rounded-3xl font-black uppercase tracking-widest hover:bg-slate-50 hover:text-slate-900 transition-all"
          >
            <ArrowLeftIcon className="w-5 h-5" />
            <span className="thai-text">Exit to Home</span>
          </Link>
        </div>
      </motion.div>
    </div>
  )
}
