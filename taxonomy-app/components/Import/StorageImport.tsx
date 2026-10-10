'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { toast } from 'react-hot-toast'
import { supabase } from '@/utils/supabase'
import {
  FolderOpenIcon,
  FileTextIcon,
  CheckCircleIcon,
  RefreshCwIcon,
  DownloadCloudIcon,
  PackageIcon,
  CalendarIcon,
  HardDriveIcon,
  SearchIcon
} from 'lucide-react'
import { FOCUS_RING } from '@/components/ui/Modal'

/* ──────────────────────────────────────────────────
   Types
────────────────────────────────────────────────── */
interface ImportRecord {
  id: string
  name: string                // ชื่อไทยต้นฉบับ
  file_name: string | null
  file_size: number | null
  file_type: string | null
  total_records: number
  processed_records: number
  success_records: number
  status: 'pending' | 'processing' | 'completed' | 'failed'
  metadata: { storage_path?: string } | null
  created_at: string
}

interface StorageImportProps {
  onFileSelect?: (file: File, fileName: string) => void
}

/* ──────────────────────────────────────────────────
   Helpers
────────────────────────────────────────────────── */
function formatBytes(bytes: number | null): string {
  if (!bytes) return '-'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`
}

function formatDate(d: string) {
  return new Date(d).toLocaleString('th-TH', {
    day: 'numeric', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit'
  })
}

const STATUS = {
  completed: { label: 'สำเร็จ',        color: 'bg-emerald-100 text-emerald-900 border border-emerald-200' },
  processing: { label: 'กำลังทำ',       color: 'bg-blue-100    text-blue-900    border border-blue-200' },
  pending:    { label: 'รอดำเนินการ',   color: 'bg-amber-100   text-amber-900   border border-amber-200' },
  failed:     { label: 'ล้มเหลว',       color: 'bg-rose-100    text-rose-900    border border-rose-200' },
}

/* ──────────────────────────────────────────────────
   Component
────────────────────────────────────────────────── */
export default function StorageImport({ onFileSelect }: StorageImportProps) {
  const [records, setRecords]         = useState<ImportRecord[]>([])
  const [filtered, setFiltered]       = useState<ImportRecord[]>([])
  const [loading, setLoading]         = useState(false)
  const [selected, setSelected]       = useState<string | null>(null)
  const [downloading, setDownloading] = useState(false)
  const [search, setSearch]           = useState('')

  useEffect(() => { loadRecords() }, [])

  useEffect(() => {
    const q = search.toLowerCase()
    setFiltered(
      q ? records.filter(r => (r.file_name || r.name || '').toLowerCase().includes(q)) : records
    )
  }, [search, records])

  /* ── ดึงจาก imports table (มีชื่อไทย + storage_path) ── */
  const loadRecords = async () => {
    setLoading(true)
    try {
      const { data, error } = await supabase
        .from('imports')
        .select('id, name, file_name, file_size, file_type, total_records, processed_records, success_records, status, metadata, created_at')
        .order('created_at', { ascending: false })
        .limit(30)

      if (error) throw error
      // กรองเฉพาะ record ที่มี storage_path (คือมีไฟล์จริงใน Storage)
      const withFile = (data || []).filter(r => r.metadata?.storage_path)
      setRecords(withFile)
      setFiltered(withFile)
    } catch (err: any) {
      console.error('Error loading import records:', err)
      toast.error('ไม่สามารถโหลดรายการได้')
    } finally {
      setLoading(false)
    }
  }

  /* ── ดาวน์โหลด + ส่งกลับ File object ── */
  const selectRecord = async (rec: ImportRecord) => {
    if (!onFileSelect) return
    const storagePath = rec.metadata?.storage_path
    if (!storagePath) { toast.error('ไม่พบไฟล์ใน Storage'); return }

    setDownloading(true)
    try {
      const { data, error } = await supabase.storage
        .from('uploads')
        .download(storagePath)

      if (error) throw error
      if (!data) throw new Error('No data')

      // ใช้ชื่อไทยต้นฉบับสำหรับ File object
      const originalName = rec.file_name || rec.name || 'import.csv'
      const file = new File([data], originalName, { type: 'text/csv' })

      toast.success(`✅ เลือกไฟล์: ${originalName}`)
      onFileSelect(file, originalName)
    } catch (err: any) {
      console.error('Download error:', err)
      toast.error(`ดาวน์โหลดไม่สำเร็จ: ${err.message}`)
    } finally {
      setDownloading(false)
    }
  }

  const displayName = (rec: ImportRecord) => rec.file_name || rec.name || 'ไม่ระบุชื่อ'

  /* ──────────────────────────────────────────────────
     Render
  ────────────────────────────────────────────────── */
  return (
    <div className="space-y-5">

      {/* ── Header ── */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-xl font-black text-slate-800 thai-text flex items-center gap-2">
            <FolderOpenIcon className="w-5 h-5 text-amber-500" aria-hidden="true" />
            เลือกไฟล์จาก Storage
          </h2>
          <p className="text-sm text-slate-600 thai-text mt-1">
            ไฟล์ที่เคยอัปโหลดแล้ว — ชื่อต้นฉบับภาษาไทยถูกเก็บไว้ครบถ้วน
          </p>
        </div>
        <button
          onClick={loadRecords}
          disabled={loading}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-sm font-bold transition-colors disabled:opacity-50 ${FOCUS_RING}`}
        >
          <RefreshCwIcon className={`w-4 h-4 ${loading ? 'motion-safe:animate-spin' : ''}`} aria-hidden="true" />
          รีเฟรช
        </button>
      </div>

      {/* ── Search ── */}
      <div className="relative">
        <label htmlFor="storage-search-input" className="sr-only">
          ค้นหาชื่อไฟล์
        </label>
        <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" aria-hidden="true" />
        <input
          id="storage-search-input"
          type="text"
          placeholder="ค้นหาชื่อไฟล์..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          className={`w-full pl-10 pr-4 py-2.5 rounded-xl border border-slate-300 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:border-indigo-500 thai-text ${FOCUS_RING}`}
        />
      </div>

      {/* ── List ── */}
      {loading ? (
        <div className="flex items-center justify-center py-16 gap-3 text-slate-600" role="status" aria-live="polite">
          <RefreshCwIcon className="w-5 h-5 motion-safe:animate-spin text-slate-500" aria-hidden="true" />
          <span className="thai-text text-sm font-medium">กำลังโหลด...</span>
        </div>
      ) : filtered.length === 0 ? (
        <div className="text-center py-16 border-2 border-dashed border-slate-200 rounded-2xl">
          <FolderOpenIcon className="w-12 h-12 mx-auto mb-3 text-slate-300" aria-hidden="true" />
          <p className="text-slate-600 font-medium thai-text text-sm">
            {search ? 'ไม่พบไฟล์ที่ตรงกับคำค้นหา' : 'ยังไม่มีไฟล์ที่อัปโหลดไว้'}
          </p>
        </div>
      ) : (
        <AnimatePresence>
          <div className="space-y-2">
            {filtered.map((rec, idx) => {
              const sConfig = STATUS[rec.status] ?? STATUS.pending
              const name    = displayName(rec)
              const isSelected = selected === rec.id

              return (
                <motion.div
                  key={rec.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.03 }}
                  role="button"
                  tabIndex={0}
                  aria-expanded={isSelected}
                  aria-label={`เลือกไฟล์ ${name}`}
                  onClick={() => setSelected(isSelected ? null : rec.id)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault()
                      setSelected(isSelected ? null : rec.id)
                    }
                  }}
                  className={`rounded-2xl border cursor-pointer transition-all overflow-hidden ${FOCUS_RING} ${
                    isSelected
                      ? 'border-indigo-400 bg-indigo-50/60 shadow-md shadow-indigo-100'
                      : 'border-slate-200 bg-white hover:border-slate-300 hover:shadow-sm'
                  }`}
                >
                  {/* ── Row ── */}
                  <div className="flex items-center gap-4 p-4">

                    {/* Icon */}
                    <div className={`w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 ${isSelected ? 'bg-indigo-600' : 'bg-slate-100'}`}>
                      <FileTextIcon className={`w-5 h-5 ${isSelected ? 'text-white' : 'text-slate-500'}`} aria-hidden="true" />
                    </div>

                    {/* Name + meta */}
                    <div className="flex-1 min-w-0">
                      <p className="font-bold text-slate-900 text-sm thai-text truncate" title={name}>
                        {name}
                      </p>
                      <div className="flex items-center gap-3 mt-0.5 text-xs text-slate-600">
                        <span className="flex items-center gap-1">
                          <CalendarIcon className="w-3 h-3 text-slate-500" aria-hidden="true" />
                          {formatDate(rec.created_at)}
                        </span>
                        <span className="flex items-center gap-1">
                          <HardDriveIcon className="w-3 h-3 text-slate-500" aria-hidden="true" />
                          {formatBytes(rec.file_size)}
                        </span>
                        <span className="flex items-center gap-1">
                          <PackageIcon className="w-3 h-3 text-slate-500" aria-hidden="true" />
                          {rec.total_records.toLocaleString()} แถว
                        </span>
                      </div>
                    </div>

                    {/* Status */}
                    <span className={`text-xs font-bold px-2.5 py-1 rounded-xl flex-shrink-0 ${sConfig.color}`}>
                      {sConfig.label}
                    </span>
                  </div>

                  {/* ── Expanded: confirm button ── */}
                  <AnimatePresence>
                    {isSelected && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.18 }}
                        className="overflow-hidden"
                      >
                        <div className="px-4 pb-4 flex items-center justify-between gap-4 border-t border-indigo-100 pt-3">
                          <p className="text-xs text-indigo-900 thai-text font-medium break-all">
                            📂 {rec.metadata?.storage_path}
                          </p>
                          <button
                            onClick={(e) => { e.stopPropagation(); selectRecord(rec) }}
                            disabled={downloading}
                            className={`flex items-center gap-2 px-5 py-2.5 bg-indigo-700 hover:bg-indigo-800 disabled:opacity-50 text-white rounded-xl font-bold text-sm transition-colors flex-shrink-0 shadow-lg shadow-indigo-200 ${FOCUS_RING}`}
                          >
                            {downloading
                              ? <><RefreshCwIcon className="w-4 h-4 motion-safe:animate-spin" aria-hidden="true" /> กำลังโหลด...</>
                              : <><DownloadCloudIcon className="w-4 h-4" aria-hidden="true" /> ใช้ไฟล์นี้</>
                            }
                          </button>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              )
            })}
          </div>
        </AnimatePresence>
      )}

      <p className="text-center text-xs text-slate-600 font-medium">
        แสดง {filtered.length} / {records.length} รายการที่มีไฟล์ใน Storage
      </p>
    </div>
  )
}
