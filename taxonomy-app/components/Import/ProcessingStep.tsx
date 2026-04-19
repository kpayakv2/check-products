'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { toast } from 'react-hot-toast'
import { supabase, DatabaseService } from '@/utils/supabase'
import {
  CheckCircleIcon,
  LoaderIcon,
  AlertCircleIcon,
  SparklesIcon,
  BrainIcon,
  LayersIcon,
  BarChart3Icon,
  AlertTriangleIcon,
  ZapIcon,
  HistoryIcon
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
  const [analytics, setAnalytics] = useState<{ avg_confidence: number; errors: number } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [importId, setImportId] = useState<string | null>(null)

  useEffect(() => {
    runFastImport()
  }, [])

  const sanitizeFileName = (name: string): string => {
    const lastDot = name.lastIndexOf('.')
    const ext = lastDot !== -1 ? name.slice(lastDot) : ''
    const base = lastDot !== -1 ? name.slice(0, lastDot) : name
    return (base.replace(/[^\w\-]/g, '_').replace(/_+/g, '_').replace(/^_|_$/g, '') || 'file') + ext
  }

  const runFastImport = async (resumeId?: string) => {
    setStatus('uploading')
    setMessage('กำลังส่งไฟล์ขึ้นคลัง...')

    try {
      let currentId = resumeId
      let path = ''

      if (!resumeId) {
        // 1. Upload
        const safeName = sanitizeFileName(file.name)
        path = `imports/${Date.now()}-${safeName}`
        const { error: upErr } = await supabase.storage.from('uploads').upload(path, file)
        if (upErr) throw upErr

        // 2. Create Batch Record
        const batch = await DatabaseService.createImport({
          name: file.name,
          file_name: file.name,
          file_size: file.size,
          file_type: 'csv',
          status: 'processing',
          total_records: parsedData.totalCount,
          metadata: { storage_path: path, last_processed_index: 0, error_log: [] }
        })
        currentId = batch.id
      } else {
        const { data } = await supabase.from('imports').select('*').eq('id', resumeId).single()
        path = data?.metadata?.storage_path
      }

      if (!currentId || !path) throw new Error('Initialization failed')
      setImportId(currentId)

      // 3. Start/Resume Streaming API
      setStatus('processing')
      const fd = new FormData()
      fd.append('filePath', path)
      fd.append('importId', currentId)
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
          try {
            const data = JSON.parse(line)
            if (data.type === 'progress') {
              setStats({ total: data.total, processed: data.processed })
              setProgress(Math.round((data.processed / data.total) * 100))
              setMessage(`กำลังทำ: ${data.last_item}`)
            } else if (data.type === 'completed') {
              setStatus('completed')
              setProgress(100)
              setAnalytics({ avg_confidence: data.avg_confidence, errors: data.errors })
              setMessage('ประมวลผลเสร็จสิ้น!')
              toast.success('วิเคราะห์ข้อมูลเสร็จเรียบร้อย')
            } else if (data.type === 'error') {
              throw new Error(data.message)
            }
          } catch (e) {
            console.error('Line parse error:', e)
          }
        }
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : String(err)
      setError(errorMessage)
      setStatus('error')
    }
  }

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Main Status Card */}
      <div className="premium-card p-12 text-center bg-white border border-slate-100 shadow-2xl shadow-indigo-500/5 rounded-[2rem] relative overflow-hidden">
        {status === 'processing' && (
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 animate-shimmer" />
        )}

        <div className={`w-24 h-24 rounded-3xl flex items-center justify-center mx-auto mb-8 transition-all duration-500 ${
          status === 'completed' ? 'bg-emerald-50 text-emerald-600 scale-110' : 
          status === 'error' ? 'bg-rose-50 text-rose-600' :
          'bg-indigo-50 text-indigo-600'
        }`}>
          {status === 'processing' ? <BrainIcon className="w-12 h-12 animate-pulse" /> :
            status === 'completed' ? <CheckCircleIcon className="w-12 h-12" /> :
            status === 'error' ? <AlertCircleIcon className="w-12 h-12" /> :
            <LoaderIcon className="w-12 h-12 animate-spin" />}
        </div>

        <h2 className="text-3xl font-black text-slate-900 mb-3 font-noto-sans-thai tracking-tight">
          {status === 'completed' ? 'AI วิเคราะห์เสร็จสิ้น!' : 
           status === 'error' ? 'เกิดข้อผิดพลาด' : 
           '🤖 พยัคฆ์ AI กำลังประมวลผล'}
        </h2>
        <p className="text-slate-400 font-bold text-sm truncate max-w-lg mx-auto uppercase tracking-widest">{message}</p>

        {/* Progress Bar */}
        <div className="mt-12 max-w-md mx-auto">
          <div className="h-4 bg-slate-50 rounded-full overflow-hidden border border-slate-100 p-1">
            <motion.div 
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }} 
              className={`h-full rounded-full ${status === 'completed' ? 'bg-emerald-500' : 'bg-indigo-600'}`} 
            />
          </div>
          <div className="mt-4 flex justify-between items-center px-1">
            <span className="text-[10px] font-black text-slate-400 uppercase tracking-tighter">Progress Tracker</span>
            <span className="text-sm font-black text-slate-900">{progress}% <span className="text-slate-300 ml-1">({stats.processed}/{stats.total})</span></span>
          </div>
        </div>
      </div>

      {/* Analytics Summary (Phase 3) */}
      <AnimatePresence>
        {status === 'completed' && analytics && (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="grid grid-cols-1 md:grid-cols-3 gap-4"
          >
            <div className="bg-white p-6 rounded-3xl border border-slate-100 shadow-sm flex items-center gap-4">
              <div className="w-12 h-12 bg-emerald-50 rounded-2xl flex items-center justify-center text-emerald-600">
                <ZapIcon className="w-6 h-6" />
              </div>
              <div>
                <div className="text-[10px] font-black text-slate-400 uppercase tracking-wider">Avg. Confidence</div>
                <div className="text-xl font-black text-slate-900">{Math.round(analytics.avg_confidence * 100)}%</div>
              </div>
            </div>

            <div className="bg-white p-6 rounded-3xl border border-slate-100 shadow-sm flex items-center gap-4">
              <div className="w-12 h-12 bg-amber-50 rounded-2xl flex items-center justify-center text-amber-600">
                <AlertTriangleIcon className="w-6 h-6" />
              </div>
              <div>
                <div className="text-[10px] font-black text-slate-400 uppercase tracking-wider">Errors/Retries</div>
                <div className="text-xl font-black text-slate-900">{analytics.errors} <span className="text-xs text-slate-300">items</span></div>
              </div>
            </div>

            <div className="bg-white p-6 rounded-3xl border border-slate-100 shadow-sm flex items-center gap-4">
              <div className="w-12 h-12 bg-indigo-50 rounded-2xl flex items-center justify-center text-indigo-600">
                <HistoryIcon className="w-6 h-6" />
              </div>
              <div>
                <div className="text-[10px] font-black text-slate-400 uppercase tracking-wider">Total Records</div>
                <div className="text-xl font-black text-slate-900">{stats.total} <span className="text-xs text-slate-300">items</span></div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Action Bar */}
      <div className="flex items-center justify-between pt-6">
        <button
          onClick={() => window.location.reload()}
          className="px-6 py-3 text-slate-400 hover:text-slate-900 font-bold text-sm transition-colors"
        >
          ยกเลิก
        </button>

        <button
          onClick={() => onComplete([])}
          disabled={status !== 'completed'}
          className={`px-12 py-4 rounded-2xl font-black text-xs uppercase tracking-[0.2em] shadow-2xl transition-all active:scale-95 ${
            status === 'completed' 
              ? 'bg-slate-900 text-white hover:bg-black shadow-slate-900/20' 
              : 'bg-slate-100 text-slate-300 cursor-not-allowed shadow-none'
          }`}
        >
          {status === 'completed' ? 'Go to Review →' : 'Analyzing Data...'}
        </button>
      </div>

      {error && (
        <div className="p-6 bg-rose-50 border border-rose-100 rounded-[2rem] flex gap-4 items-start animate-shake">
          <AlertCircleIcon className="w-6 h-6 text-rose-600 flex-shrink-0" />
          <div className="flex-1">
            <div className="text-rose-900 font-black text-sm uppercase mb-1">System Error Detected</div>
            <div className="text-rose-600/80 text-xs font-medium font-mono">{error}</div>
            <button 
              onClick={() => runFastImport(importId || undefined)}
              className="mt-4 px-4 py-2 bg-rose-600 text-white rounded-xl text-[10px] font-black uppercase tracking-widest hover:bg-rose-700 transition-colors"
            >
              Retry Operation
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
