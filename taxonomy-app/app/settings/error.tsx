'use client'

import { useEffect } from 'react'
import { motion } from 'framer-motion'
import { Settings, RefreshCcw, Home } from 'lucide-react'
import Link from 'next/link'

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  useEffect(() => {
    console.error('Settings Error:', error)
  }, [error])

  return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center p-6 relative overflow-hidden">
      {/* Decorative elements */}
      <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-slate-200/50 rounded-full blur-[100px] -mr-48 -mt-48 pointer-events-none" />
      <div className="absolute bottom-10 left-10 w-[300px] h-[300px] bg-slate-100/30 rounded-full blur-[100px] pointer-events-none" />
      
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-md w-full bg-white/80 backdrop-blur-xl rounded-[48px] shadow-2xl p-12 border border-white/20 relative z-10 text-center"
      >
        <div className="w-24 h-24 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-8 border border-slate-200 shadow-inner">
           <Settings className="w-10 h-10 text-slate-500" />
        </div>

        <h2 className="text-2xl font-black text-slate-900 mb-2 uppercase tracking-tight">
          Settings Configuration Error
        </h2>
        <h3 className="text-xl font-bold text-slate-700 mb-4 thai-text">
          ข้อผิดพลาดในการตั้งค่าระบบ
        </h3>
        
        <p className="text-slate-500 mb-10 thai-text leading-relaxed">
          ไม่สามารถบันทึกหรือดึงข้อมูลการตั้งค่าได้ในขณะนี้ กรุณาลองใหม่อีกครั้ง หรือตรวจสอบการเชื่อมต่อครับ
        </p>

        <div className="flex flex-col gap-4">
          <button
            onClick={() => reset()}
            className="w-full flex items-center justify-center gap-3 bg-slate-800 text-white py-5 rounded-[24px] font-black uppercase tracking-widest hover:bg-slate-900 transition-all shadow-xl shadow-slate-200 group"
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
             Error ID: {error.digest || 'SETTINGS_ERR_UNKNOWN'}
           </p>
        </div>
      </motion.div>
    </div>
  )
}
