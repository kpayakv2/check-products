'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { toast } from 'react-hot-toast'
import { supabase, DatabaseService } from '@/utils/supabase'
import {
  CheckCircleIcon,
  LoaderIcon,
  AlertCircleIcon,
  SparklesIcon,
  BrainIcon,
  LayersIcon
} from 'lucide-react'
import type { ParsedCSV } from '@/utils/csv-parser'
import type { ColumnMapping } from './ColumnMappingStep'

interface ProcessingStepProps {
  file: File
  columnMapping: ColumnMapping
  parsedData: ParsedCSV
  onComplete: (results: any[]) => void
  onBack?: () => void
}

export default function ProcessingStep({
  file,
  columnMapping,
  parsedData,
  onComplete
}: ProcessingStepProps) {
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState<'idle' | 'uploading' | 'processing' | 'completed' | 'error'>('idle')
  const [message, setMessage] = useState('เตรียมตัว...')
  const [stats, setStats] = useState({ total: parsedData.totalCount, processed: 0 })
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    runFastImport()
  }, [])

  /** Sanitize filename: replace Thai/unicode chars with transliteration-safe ASCII */
  const sanitizeFileName = (name: string): string => {
    // Split extension
    const lastDot = name.lastIndexOf('.')
    const ext = lastDot !== -1 ? name.slice(lastDot) : ''
    const base = lastDot !== -1 ? name.slice(0, lastDot) : name

    // Replace any character that is NOT alphanumeric, hyphen, or underscore
    const safe = base
      .replace(/[^\w\-]/g, '_')   // non-word chars → underscore
      .replace(/_+/g, '_')         // collapse multiple underscores
      .replace(/^_|_$/g, '')       // trim leading/trailing underscores

    return (safe || 'file') + ext
  }

  const runFastImport = async () => {
    setStatus('uploading')
    setMessage('กำลังส่งไฟล์ขึ้นคลัง...')
    
    try {
      // 1. Upload (sanitize filename to ASCII to avoid Supabase Storage "Invalid key" error)
      const safeName = sanitizeFileName(file.name)
      const path = `imports/${Date.now()}-${safeName}`
      const { error: upErr } = await supabase.storage.from('uploads').upload(path, file)
      if (upErr) throw upErr

      // 2. Create Batch (preserve original Thai filename in file_name field)
      const batch = await DatabaseService.createImport({
        name: file.name,                          // ชื่อไฟล์ภาษาไทยต้นฉบับ
        file_name: file.name,                     // สำรองไว้ใน field เฉพาะ
        file_size: file.size,
        file_type: 'csv',
        status: 'processing',
        total_records: parsedData.totalCount,
        metadata: { storage_path: path }          // path จริงใน Supabase Storage
      })

      // 3. Start Streaming API
      setStatus('processing')
      const fd = new FormData()
      fd.append('filePath', path)
      fd.append('importId', batch.id)
      fd.append('columnMapping', JSON.stringify(columnMapping))

      const res = await fetch('/api/import/process', { method: 'POST', body: fd })
      if (!res.ok) throw new Error('API Fail')

      const reader = res.body?.getReader()
      if (!reader) throw new Error('No stream')

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.trim()) continue
          const data = JSON.parse(line)
          
          if (data.type === 'progress') {
            setStats({ total: data.total, processed: data.processed })
            setProgress(Math.round((data.processed / data.total) * 100))
            setMessage(`กำลังทำ: ${data.last_item}`)
          } else if (data.type === 'completed') {
            setStatus('completed')
            setProgress(100)
            setMessage('ประมวลผลเสร็จสิ้น!')
            toast.success('เรียบร้อย!')
          } else if (data.type === 'error') {
            throw new Error(data.message)
          }
        }
      }
    } catch (err) {
      setError(err.message)
      setStatus('error')
    }
  }

  return (
    <div className="space-y-8">
      <div className="premium-card p-10 text-center">
        <div className={`w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-6 ${
          status === 'completed' ? 'bg-emerald-100' : 'bg-blue-50'
        }`}>
          {status === 'processing' ? <BrainIcon className="w-10 h-10 text-blue-600 animate-pulse" /> : 
           status === 'completed' ? <CheckCircleIcon className="w-10 h-10 text-emerald-600" /> :
           <LoaderIcon className="w-10 h-10 text-blue-400 animate-spin" />}
        </div>
        <h2 className="text-2xl font-black text-slate-900 mb-2 font-noto-sans-thai">
          {status === 'completed' ? 'สำเร็จ!' : '🤖 AI กำลังทำงาน'}
        </h2>
        <p className="text-slate-500 font-medium truncate max-w-md mx-auto">{message}</p>
        
        <div className="mt-8 max-w-xs mx-auto h-2 bg-slate-100 rounded-full overflow-hidden border">
          <motion.div animate={{ width: `${progress}%` }} className="h-full bg-indigo-600" />
        </div>
        <p className="mt-2 text-[10px] font-black text-slate-400">{progress}% ({stats.processed}/{stats.total})</p>
      </div>

      <div className="flex justify-end pt-4">
        <button
          onClick={() => onComplete([])}
          disabled={status !== 'completed'}
          data-testid="wizard-final-next-btn"
          className={`px-10 py-4 rounded-2xl font-black text-sm shadow-xl transition-all ${
            status === 'completed' ? 'bg-emerald-600 text-white hover:scale-105' : 'bg-slate-200 text-slate-400'
          }`}
        >
          {status === 'completed' ? 'ถัดไป: ตรวจสอบ →' : 'กำลังประมวลผล...'}
        </button>
      </div>
      {error && <div className="p-4 bg-rose-50 text-rose-600 rounded-xl text-xs font-bold border border-rose-100">{error}</div>}
    </div>
  )
}
