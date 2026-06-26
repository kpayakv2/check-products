'use client'

import { useEffect } from 'react'
import { motion } from 'framer-motion'
import { Database, RefreshCcw, Home } from 'lucide-react'
import Link from 'next/link'

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  useEffect(() => {
    console.error('Products Error:', error)
  }, [error])

  return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center p-6 relative overflow-hidden">
      {/* Decorative elements */}
      <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-emerald-50/50 rounded-full blur-[100px] -mr-48 -mt-48 pointer-events-none" />
      <div className="absolute bottom-10 left-10 w-[300px] h-[300px] bg-indigo-50/30 rounded-full blur-[100px] pointer-events-none" />
      
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-md w-full bg-white/80 backdrop-blur-xl rounded-[48px] shadow-2xl p-12 border border-white/20 relative z-10 text-center"
      >
        <div className="w-24 h-24 bg-emerald-50 rounded-full flex items-center justify-center mx-auto mb-8 border border-emerald-100 shadow-inner">
           <Database className="w-10 h-10 text-emerald-500" />
        </div>

        <h2 className="text-2xl font-black text-slate-900 mb-2 uppercase tracking-tight">
          Products Management Error
        </h2>
        <h3 className="text-xl font-bold text-slate-700 mb-4 thai-text">
          ข้อผิดพลาดในการจัดการข้อมูลสินค้า
        </h3>
        
        <p className="text-slate-500 mb-10 thai-text leading-relaxed">
          ไม่สามารถเข้าถึงหรือจัดการข้อมูลสินค้าได้ในขณะนี้ กรุณาลองใหม่อีกครั้งหรือติดต่อผู้ดูแลระบบครับ
        </p>

        <div className="flex flex-col gap-4">
          <button
            onClick={() => reset()}
            className="w-full flex items-center justify-center gap-3 bg-emerald-600 text-white py-5 rounded-[24px] font-black uppercase tracking-widest hover:bg-emerald-700 transition-all shadow-xl shadow-emerald-100 group"
          >
            <RefreshCcw className="w-5 h-5 group-hover:rotate-180 transition-transform duration-500" />
            <span className="thai-text">พยายามอีกครั้ง</span>
          </button>
          
          <Link 
            href="/"
            className="w-full flex items-center justify-center gap-3 bg-slate-100 text-slate-500 py-5 rounded-[24px] font-black uppercase tracking-widest hover:bg-slate-200 hover:text-slate-900 transition-all"
          >
            <Home className="w-5 h-5" />
            <span className="thai-text">กลับหน้าหลัก</span>
          </Link>
        </div>

        <div className="mt-10 pt-10 border-t border-slate-100/50">
           <p className="text-[10px] font-bold text-slate-300 uppercase tracking-[0.3em]">
             Error ID: {error.digest || 'PRODUCT_ERR_UNKNOWN'}
           </p>
        </div>
      </motion.div>
    </div>
  )
}
