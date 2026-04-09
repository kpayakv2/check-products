'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { supabase } from '@/utils/supabase'
import {
  FileTextIcon,
  DownloadIcon,
  CheckCircleIcon,
  ClockIcon,
  AlertCircleIcon,
  RefreshCwIcon,
  DatabaseIcon,
  PackageIcon,
  ChevronDownIcon,
  ChevronUpIcon
} from 'lucide-react'

interface ImportRecord {
  id: string
  name: string
  file_name: string | null
  file_size: number | null
  file_type: string | null
  total_records: number
  processed_records: number
  success_records: number
  error_records: number
  status: 'pending' | 'processing' | 'completed' | 'failed'
  metadata: { storage_path?: string } | null
  created_at: string
  completed_at: string | null
}

const STATUS_CONFIG = {
  completed: { label: 'สำเร็จ', icon: CheckCircleIcon, color: 'text-emerald-600', bg: 'bg-emerald-50', border: 'border-emerald-100' },
  processing: { label: 'กำลังทำ', icon: ClockIcon, color: 'text-blue-600', bg: 'bg-blue-50', border: 'border-blue-100' },
  pending:    { label: 'รอดำเนินการ', icon: ClockIcon, color: 'text-amber-600', bg: 'bg-amber-50', border: 'border-amber-100' },
  failed:     { label: 'ล้มเหลว', icon: AlertCircleIcon, color: 'text-rose-600', bg: 'bg-rose-50', border: 'border-rose-100' },
}

function formatBytes(bytes: number | null): string {
  if (!bytes) return '-'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`
}

function formatDate(dateStr: string): string {
  return new Date(dateStr).toLocaleString('th-TH', {
    year: 'numeric', month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit'
  })
}

export default function ImportHistory() {
  const [records, setRecords] = useState<ImportRecord[]>([])
  const [loading, setLoading] = useState(false)
  const [expanded, setExpanded] = useState<string | null>(null)
  const [downloading, setDownloading] = useState<string | null>(null)

  useEffect(() => { loadHistory() }, [])

  const loadHistory = async () => {
    setLoading(true)
    try {
      const { data, error } = await supabase
        .from('imports')
        .select('*')
        .order('created_at', { ascending: false })
        .limit(20)
      if (error) throw error
      setRecords(data || [])
    } catch (err) {
      console.error('Error loading import history:', err)
    } finally {
      setLoading(false)
    }
  }

  const downloadFile = async (record: ImportRecord) => {
    const path = record.metadata?.storage_path
    if (!path) return alert('ไม่พบ storage path สำหรับไฟล์นี้')

    setDownloading(record.id)
    try {
      const { data, error } = await supabase.storage
        .from('uploads')
        .download(path)
      if (error) throw error
      if (!data) throw new Error('No file data')

      // Download as original Thai filename
      const url = URL.createObjectURL(data)
      const a = document.createElement('a')
      a.href = url
      a.download = record.file_name || record.name || 'export.csv'
      a.click()
      URL.revokeObjectURL(url)
    } catch (err: any) {
      alert(`ดาวน์โหลดไม่สำเร็จ: ${err.message}`)
    } finally {
      setDownloading(null)
    }
  }

  if (loading) return (
    <div className="flex items-center justify-center py-16 gap-3 text-slate-400">
      <RefreshCwIcon className="w-5 h-5 animate-spin" />
      <span className="font-medium thai-text">กำลังโหลดประวัติ...</span>
    </div>
  )

  if (records.length === 0) return (
    <div className="text-center py-16">
      <DatabaseIcon className="w-12 h-12 mx-auto mb-3 text-slate-200" />
      <p className="text-slate-400 font-medium thai-text">ยังไม่มีประวัติการนำเข้า</p>
    </div>
  )

  return (
    <div className="space-y-3">
      <AnimatePresence>
        {records.map((rec, idx) => {
          const cfg = STATUS_CONFIG[rec.status] ?? STATUS_CONFIG.pending
          const StatusIcon = cfg.icon
          const isExpanded = expanded === rec.id
          const displayName = rec.file_name || rec.name || 'ไฟล์ไม่ระบุชื่อ'
          const hasStorage = !!rec.metadata?.storage_path

          return (
            <motion.div
              key={rec.id}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ delay: idx * 0.04 }}
              className="bg-white rounded-2xl border border-slate-100 overflow-hidden hover:border-slate-200 transition-colors"
            >
              {/* Header Row */}
              <div
                className="flex items-center gap-4 p-4 cursor-pointer"
                onClick={() => setExpanded(isExpanded ? null : rec.id)}
              >
                {/* File icon */}
                <div className="w-10 h-10 rounded-xl bg-slate-50 flex items-center justify-center flex-shrink-0">
                  <FileTextIcon className="w-5 h-5 text-slate-400" />
                </div>

                {/* Name + date */}
                <div className="flex-1 min-w-0">
                  <p className="font-bold text-slate-800 text-sm thai-text truncate" title={displayName}>
                    {displayName}
                  </p>
                  <p className="text-xs text-slate-400 font-medium mt-0.5">
                    {formatDate(rec.created_at)} · {formatBytes(rec.file_size)}
                  </p>
                </div>

                {/* Status badge */}
                <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl border text-xs font-black ${cfg.color} ${cfg.bg} ${cfg.border} flex-shrink-0`}>
                  <StatusIcon className="w-3.5 h-3.5" />
                  {cfg.label}
                </div>

                {/* Stats */}
                <div className="hidden md:flex items-center gap-1 text-xs text-slate-400 flex-shrink-0">
                  <PackageIcon className="w-3.5 h-3.5" />
                  <span className="font-bold text-slate-600">{rec.success_records ?? 0}</span>
                  <span>/ {rec.total_records} รายการ</span>
                </div>

                {/* Download */}
                {hasStorage && (
                  <button
                    onClick={(e) => { e.stopPropagation(); downloadFile(rec) }}
                    disabled={downloading === rec.id}
                    title={`ดาวน์โหลด: ${displayName}`}
                    className="p-2 rounded-xl hover:bg-indigo-50 text-slate-400 hover:text-indigo-600 transition-colors flex-shrink-0 disabled:opacity-40"
                  >
                    {downloading === rec.id
                      ? <RefreshCwIcon className="w-4 h-4 animate-spin" />
                      : <DownloadIcon className="w-4 h-4" />
                    }
                  </button>
                )}

                {/* Expand toggle */}
                <div className="text-slate-300 flex-shrink-0">
                  {isExpanded
                    ? <ChevronUpIcon className="w-4 h-4" />
                    : <ChevronDownIcon className="w-4 h-4" />}
                </div>
              </div>

              {/* Expanded detail */}
              <AnimatePresence>
                {isExpanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                  >
                    <div className="px-4 pb-4 pt-0 border-t border-slate-50 bg-slate-50/50">
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
                        <div className="bg-white rounded-xl p-3 border border-slate-100">
                          <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">ชื่อไฟล์ต้นฉบับ</p>
                          <p className="text-xs font-bold text-slate-700 thai-text break-all">{displayName}</p>
                        </div>
                        <div className="bg-white rounded-xl p-3 border border-slate-100">
                          <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">Storage Path</p>
                          <p className="text-xs font-mono text-slate-500 break-all">{rec.metadata?.storage_path ?? '-'}</p>
                        </div>
                        <div className="bg-white rounded-xl p-3 border border-slate-100">
                          <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">สำเร็จ / ทั้งหมด</p>
                          <p className="text-sm font-black text-emerald-600">{rec.success_records ?? 0}
                            <span className="text-slate-400 font-medium text-xs"> / {rec.total_records}</span>
                          </p>
                        </div>
                        <div className="bg-white rounded-xl p-3 border border-slate-100">
                          <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">ข้อผิดพลาด</p>
                          <p className={`text-sm font-black ${(rec.error_records ?? 0) > 0 ? 'text-rose-500' : 'text-slate-300'}`}>
                            {rec.error_records ?? 0} รายการ
                          </p>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          )
        })}
      </AnimatePresence>
    </div>
  )
}
