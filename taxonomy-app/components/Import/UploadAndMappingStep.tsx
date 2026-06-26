'use client'

import { motion } from 'framer-motion'
import { UploadIcon, FolderIcon, CheckCircleIcon } from 'lucide-react'
import StorageImport from './StorageImport'
import ColumnMappingStep from './ColumnMappingStep'
import { toast } from 'react-hot-toast'

export default function UploadAndMappingStep({
  file,
  setFile,
  importMode,
  setImportMode,
  onComplete
}: any) {
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = event.target.files?.[0]
    if (uploadedFile) {
      setFile(uploadedFile)
    }
  }

  if (file) {
    return (
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
        <ColumnMappingStep 
          file={file} 
          onComplete={onComplete} 
          onBack={() => setFile(null)} 
        />
      </motion.div>
    )
  }

  return (
    <div className="space-y-8">
      {/* Import Mode Selection */}
      <div className="bg-slate-900/60 backdrop-blur-xl border border-white/10 rounded-3xl p-8 shadow-2xl relative overflow-hidden group">
        <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-500/10 rounded-full blur-[80px] -mr-32 -mt-32 transition-transform group-hover:scale-110" />
        <h2 className="text-2xl font-bold mb-6 thai-text text-white flex items-center gap-3">
          <div className="w-10 h-10 bg-indigo-500/20 rounded-xl flex items-center justify-center border border-indigo-500/30">
            <UploadIcon className="w-5 h-5 text-indigo-400" />
          </div>
          เลือกแหล่งข้อมูล (Data Source)
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 relative z-10">
          <button
            onClick={() => setImportMode('upload')}
            className={`p-6 border rounded-2xl transition-all flex flex-col items-start relative overflow-hidden ${
              importMode === 'upload' 
                ? 'border-indigo-500 bg-indigo-500/10 shadow-[0_0_30px_rgba(99,102,241,0.15)]' 
                : 'border-white/10 bg-white/5 hover:bg-white/10 hover:border-white/20'
            }`}
          >
            {importMode === 'upload' && <div className="absolute top-0 left-0 w-1 h-full bg-indigo-500 shadow-[0_0_10px_rgba(99,102,241,1)]" />}
            <UploadIcon className={`w-8 h-8 mb-4 transition-colors ${importMode === 'upload' ? 'text-indigo-400' : 'text-slate-400'}`} />
            <h3 className={`text-lg font-bold mb-2 thai-text tracking-wide ${importMode === 'upload' ? 'text-white' : 'text-slate-300'}`}>อัปโหลดไฟล์ใหม่จากเครื่อง</h3>
            <p className="text-sm text-slate-500 thai-text text-left">ลากและวางไฟล์ CSV จากคอมพิวเตอร์ของคุณเพื่อเริ่มกระบวนการ</p>
          </button>
          <button
            onClick={() => setImportMode('storage')}
            className={`p-6 border rounded-2xl transition-all flex flex-col items-start relative overflow-hidden ${
              importMode === 'storage' 
                ? 'border-emerald-500 bg-emerald-500/10 shadow-[0_0_30px_rgba(16,185,129,0.15)]' 
                : 'border-white/10 bg-white/5 hover:bg-white/10 hover:border-white/20'
            }`}
          >
            {importMode === 'storage' && <div className="absolute top-0 left-0 w-1 h-full bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,1)]" />}
            <FolderIcon className={`w-8 h-8 mb-4 transition-colors ${importMode === 'storage' ? 'text-emerald-400' : 'text-slate-400'}`} />
            <h3 className={`text-lg font-bold mb-2 thai-text tracking-wide ${importMode === 'storage' ? 'text-white' : 'text-slate-300'}`}>ดึงข้อมูลจาก Cloud Storage</h3>
            <p className="text-sm text-slate-500 thai-text text-left">เลือกไฟล์ที่เคยอัปโหลดไว้แล้วในระบบ Supabase Storage</p>
          </button>
        </div>
      </div>

      {/* Upload Mode */}
      {importMode === 'upload' && (
        <motion.div 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-slate-900/60 backdrop-blur-xl border border-white/10 rounded-3xl p-8 shadow-2xl"
        >
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-xl font-bold thai-text text-white">
              อัปโหลดไฟล์ (Upload CSV)
            </h3>
            <span className="text-xs font-mono text-indigo-400 bg-indigo-500/10 px-3 py-1 rounded-full border border-indigo-500/20">
              APPROVED_PRODUCTS_*.CSV
            </span>
          </div>

          <div className="border-2 border-dashed border-indigo-500/30 rounded-2xl p-16 text-center bg-indigo-500/5 hover:bg-indigo-500/10 transition-all relative group overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity" />
            
            <div className="relative z-10 flex flex-col items-center">
              <div className="w-20 h-20 bg-slate-900 rounded-full flex items-center justify-center mb-6 shadow-xl border border-white/5 group-hover:scale-110 group-hover:border-indigo-500/50 transition-all">
                <UploadIcon className="h-8 w-8 text-indigo-400 group-hover:text-indigo-300" />
              </div>
            
              <div className="space-y-3">
                <p className="text-xl font-medium text-slate-300 thai-text">
                  ลากไฟล์มาวางที่นี่ หรือ <span className="text-indigo-400 font-bold underline underline-offset-4 decoration-indigo-400/30 cursor-pointer">คลิกเพื่อเบราว์เซอร์</span>
                </p>
                <p className="text-sm text-slate-500 font-mono">
                  Supports .CSV, .XLSX up to 50MB
                </p>
              </div>
            </div>
            
            <input
              type="file"
              accept=".csv,.xlsx"
              onChange={handleFileUpload}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20"
            />
          </div>
        </motion.div>
      )}

      {/* Storage Mode */}
      {importMode === 'storage' && (
        <motion.div 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          <div className="bg-slate-900/60 backdrop-blur-xl border border-white/10 rounded-3xl p-8 shadow-2xl">
            <StorageImport 
              onFileSelect={(selectedFile: any, fileName: string) => {
                setFile(selectedFile)
                toast.success(`เลือกไฟล์: ${fileName}`)
              }}
            />
          </div>
        </motion.div>
      )}
    </div>
  )
}
