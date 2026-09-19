'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import {
  AlertCircle,
  AlertTriangle,
  Check,
  CheckCircle2,
  Download,
  FileText,
  Hash,
  Loader2,
  RefreshCcw,
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import type { PosExportIssues, PosExportRow } from '@/utils/pos-export'

interface ExportPreview {
  total: number
  issues: PosExportIssues
  preview: PosExportRow[]
}

/** สิ่งที่ค้างไว้หลังโหลดสำเร็จ — การ์ดสรุปต้องอยู่ต่อจนคนกดปิดเอง */
interface ExportResult {
  total: number
  fileName: string
  exportedAt: Date
  blob: Blob
  missingPrice: number
}

/** ช่องบังคับทั้ง 6 ของ POS — ขาดช่องไหน POS ไม่รับไฟล์ */
const MANDATORY_FIELDS: { key: keyof PosExportIssues; label: string }[] = [
  { key: 'missingItemId', label: 'รหัสสินค้า' },
  { key: 'missingBarcode', label: 'บาร์โค้ด' },
  { key: 'missingItemName', label: 'ชื่อสินค้า' },
  { key: 'missingUnitName', label: 'ชื่อหน่วย' },
  { key: 'missingItemGroup', label: 'กลุ่มสินค้า' },
  { key: 'missingPrice', label: 'ราคาขาย' },
]

const NO_ISSUES: PosExportIssues = {
  missingItemId: 0, missingBarcode: 0, missingItemName: 0,
  missingUnitName: 0, missingItemGroup: 0, missingPrice: 0,
}

/** คนเดิมกดส่งออกหลายรอบติดกัน ไม่ต้องถามซ้ำทุกรอบในแท็บเบราว์เซอร์เดียวกัน */
const SKIP_CONFIRM_KEY = 'pos-export-skip-confirm'

function readSkipConfirm(): boolean {
  try {
    return window.sessionStorage.getItem(SKIP_CONFIRM_KEY) === '1'
  } catch {
    // โหมดส่วนตัวของบางเบราว์เซอร์โยน error ตอนอ่าน storage — ถามต่อไปดีกว่าพัง
    return false
  }
}

function writeSkipConfirm(value: boolean): void {
  try {
    if (value) window.sessionStorage.setItem(SKIP_CONFIRM_KEY, '1')
    else window.sessionStorage.removeItem(SKIP_CONFIRM_KEY)
  } catch {
    /* เขียนไม่ได้ก็แค่ถามใหม่รอบหน้า ไม่ต้องแจ้งคน */
  }
}

function pad(value: number): string {
  return String(value).padStart(2, '0')
}

/** pos-import-YYYYMMDD-HHmm.txt — ชื่อสำรองเมื่ออ่านจากหัว response ไม่ได้ */
function fallbackFileName(now: Date): string {
  return `pos-import-${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(
    now.getHours()
  )}${pad(now.getMinutes())}.txt`
}

/**
 * ชื่อไฟล์จริงมาจาก Content-Disposition ของ API
 * แต่ห้ามพึ่งว่ามันมีเสมอ — response ที่ถูกจำลองในเทสต์ไม่มี headers เลย
 */
function resolveFileName(response: Response, now: Date): string {
  const headers = response.headers as Headers | undefined
  const raw = typeof headers?.get === 'function' ? headers.get('content-disposition') : null
  const match = raw?.match(/filename\*?=(?:UTF-8'')?"?([^";]+)"?/i)
  return match?.[1] ? decodeURIComponent(match[1]) : fallbackFileName(now)
}

function triggerDownload(blob: Blob, fileName: string): void {
  // ต้องผ่าน blob เพราะ response มาจาก POST ที่พาคุกกี้ไปด้วย
  // จะใช้ <a href> ตรง ๆ ไม่ได้ — นั่นเป็น GET
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = fileName
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

/** อ่านค่า prefers-reduced-motion ตรง ๆ จาก matchMedia — ไม่พึ่ง hook ของ framer-motion */
function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false)

  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return
    // เบราว์เซอร์เก่าบางตัว (และ mock ในเทสต์) คืน undefined — อย่าเชื่อว่าได้ object เสมอ
    const query = window.matchMedia('(prefers-reduced-motion: reduce)') as MediaQueryList | undefined
    if (!query) return
    setReduced(Boolean(query.matches))
    const onChange = (event: MediaQueryListEvent) => setReduced(event.matches)
    query.addEventListener?.('change', onChange)
    return () => query.removeEventListener?.('change', onChange)
  }, [])

  return reduced
}

const FOCUSABLE = 'button:not([disabled]), input:not([disabled]), a[href], [tabindex]:not([tabindex="-1"])'

const FOCUS_RING = 'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2'

/**
 * ส่งออกสินค้าใหม่ที่ยังไม่มีในสตอก เป็นไฟล์ .txt ที่ POS นำเข้าได้
 *
 * เตือนจำนวนแถวที่ยังไม่มีรหัสสินค้า/ราคาก่อนโหลด เพราะสองช่องนั้น
 * ยังว่างทั้งตารางจนกว่าจะ map คอลัมน์ตอน import รอบหน้า
 * (ดู docs/specs/2026-09-06-pos-export.md หัวข้อข้อจำกัด)
 *
 * POST เขียนฐานข้อมูลถาวร (ออกรหัสสินค้า + ตีว่าส่งออกแล้ว) และย้อนไม่ได้
 * จึงมีกล่องยืนยันคั่นไว้อีกชั้น ปุ่ม pos-export-download เปิดกล่อง
 * ส่วนปุ่มที่ยิง POST จริงคือ pos-export-confirm
 */
export default function PosExportTab() {
  const [data, setData] = useState<ExportPreview | null>(null)
  const [loading, setLoading] = useState(true)
  const [downloading, setDownloading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [dontAskAgain, setDontAskAgain] = useState(false)
  const [result, setResult] = useState<ExportResult | null>(null)

  const prefersReducedMotion = usePrefersReducedMotion()
  const downloadButtonRef = useRef<HTMLButtonElement | null>(null)
  const dialogRef = useRef<HTMLDivElement | null>(null)
  const successHeadingRef = useRef<HTMLHeadingElement | null>(null)

  // ส่งออกสำเร็จแล้วปุ่มดาวน์โหลดถูกถอดออกจาก DOM คืนโฟกัสไปที่นั่นไม่ได้
  // ย้ายไปไว้ที่หัวการ์ดสรุปแทน คนใช้คีย์บอร์ดจะได้ไม่ตกไปอยู่ที่ body
  useEffect(() => {
    if (result) successHeadingRef.current?.focus()
  }, [result])

  const fetchPreview = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch('/api/pos-export')
      const payload = await response.json()
      if (!payload.success) throw new Error(payload.error || 'โหลดข้อมูลไม่สำเร็จ')
      setData({ total: payload.total, issues: payload.issues, preview: payload.preview })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'โหลดข้อมูลไม่สำเร็จ')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void fetchPreview()
  }, [fetchPreview])

  /** คืน true เมื่อได้ไฟล์จริง — ผู้เรียกใช้ตัดสินใจเรื่องโฟกัสต่อจากนี้ */
  const runExport = useCallback(async (): Promise<boolean> => {
    setDownloading(true)
    setError(null)
    try {
      const response = await fetch('/api/pos-export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mark_exported: true }),
      })

      if (!response.ok) {
        const payload = await response.json().catch(() => null)
        throw new Error(payload?.error || 'สร้างไฟล์ไม่สำเร็จ')
      }

      const blob = await response.blob()
      const now = new Date()
      const fileName = resolveFileName(response, now)
      triggerDownload(blob, fileName)

      // ไม่โหลดใหม่ทันที — การ์ดสรุปต้องค้างให้อ่าน ไม่ใช่เด้งไปหน้าว่างเปล่า
      setResult({
        total: data?.total ?? 0,
        fileName,
        exportedAt: now,
        blob,
        missingPrice: data?.issues.missingPrice ?? 0,
      })
      setData(null)
      return true
    } catch (err) {
      setError(err instanceof Error ? err.message : 'สร้างไฟล์ไม่สำเร็จ')
      return false
    } finally {
      setDownloading(false)
    }
  }, [data])

  const closeConfirm = useCallback(() => {
    setConfirmOpen(false)
    downloadButtonRef.current?.focus()
  }, [])

  const handleDownloadClick = useCallback(() => {
    if (readSkipConfirm()) {
      void runExport()
      return
    }
    setConfirmOpen(true)
  }, [runExport])

  const handleConfirm = useCallback(async () => {
    writeSkipConfirm(dontAskAgain)
    setConfirmOpen(false)
    const exported = await runExport()
    // สำเร็จ: effect ด้านบนย้ายโฟกัสไปที่หัวการ์ดสรุปแล้ว
    // ล้มเหลว: หน้ายังเป็นหน้าเดิม ปุ่มดาวน์โหลดยังอยู่ คืนโฟกัสให้ปุ่มนั้น
    if (!exported) downloadButtonRef.current?.focus()
  }, [dontAskAgain, runExport])

  // Escape ปิดกล่อง และ Tab วนอยู่ในกล่องเท่านั้น
  useEffect(() => {
    if (!confirmOpen) return

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        closeConfirm()
        return
      }
      if (event.key !== 'Tab' || !dialogRef.current) return

      const nodes = Array.from(dialogRef.current.querySelectorAll<HTMLElement>(FOCUSABLE))
      if (nodes.length === 0) return
      const first = nodes[0]
      const last = nodes[nodes.length - 1]
      const active = document.activeElement

      if (event.shiftKey && (active === first || !dialogRef.current.contains(active))) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && active === last) {
        event.preventDefault()
        first.focus()
      }
    }

    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [confirmOpen, closeConfirm])

  if (loading) {
    return (
      <div
        data-testid="pos-export-loading"
        className="flex flex-col items-center justify-center gap-3 py-16 bg-slate-50 border border-slate-200 rounded-xl text-slate-500 thai-text"
      >
        <Loader2 className="w-6 h-6 animate-spin text-indigo-500" />
        <p className="text-sm font-bold">กำลังโหลดข้อมูลส่งออก...</p>
      </div>
    )
  }

  // ---------- State 5: สรุปหลังส่งออกสำเร็จ ----------
  if (result) {
    return (
      <div data-testid="pos-export-success" className="thai-text space-y-4">
        <div className="p-6 bg-white border border-emerald-200 rounded-xl">
          <div className="flex items-start gap-4">
            <span className="shrink-0 flex items-center justify-center w-12 h-12 rounded-full bg-emerald-50">
              <CheckCircle2 className="w-7 h-7 text-emerald-600" />
            </span>
            <div className="min-w-0">
              <h3
                ref={successHeadingRef}
                tabIndex={-1}
                className={`text-lg font-black text-slate-800 rounded-lg ${FOCUS_RING}`}
              >
                ส่งออกเรียบร้อย {result.total.toLocaleString()} รายการ
              </h3>
              <p className="text-sm text-slate-500">
                คิวหน้านี้ว่างแล้ว รายการทั้งหมดถูกตีว่าส่งออกแล้ว
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-5">
            <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg min-w-0">
              <p className="text-xs font-bold text-slate-400">ชื่อไฟล์</p>
              <p className="text-xs font-bold text-slate-700 break-all">{result.fileName}</p>
            </div>
            <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
              <p className="text-xs font-bold text-slate-400">จำนวนแถว</p>
              <p className="text-xs font-bold text-slate-700 tabular-nums">
                {result.total.toLocaleString()} แถว
              </p>
            </div>
            <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
              <p className="text-xs font-bold text-slate-400">เวลาที่ส่งออก</p>
              <p className="text-xs font-bold text-slate-700 tabular-nums">
                {pad(result.exportedAt.getHours())}:{pad(result.exportedAt.getMinutes())} น.
              </p>
            </div>
          </div>
        </div>

        <div className="p-5 bg-white border border-slate-200 rounded-xl">
          <h4 className="text-sm font-black text-slate-800 mb-3">ขั้นตอนถัดไป</h4>
          <ol className="space-y-2 text-xs text-slate-600">
            <li className="flex gap-2">
              <span className="shrink-0 w-5 h-5 rounded-lg bg-slate-100 text-slate-600 font-black flex items-center justify-center tabular-nums">
                1
              </span>
              <span>เปิดโปรแกรม POS แล้วไปที่เมนูนำเข้าสินค้า</span>
            </li>
            <li className="flex gap-2">
              <span className="shrink-0 w-5 h-5 rounded-lg bg-slate-100 text-slate-600 font-black flex items-center justify-center tabular-nums">
                2
              </span>
              <span>เลือกไฟล์ที่เพิ่งโหลด แล้วสั่งนำเข้า</span>
            </li>
            {result.missingPrice > 0 && (
              <li className="flex gap-2 p-2 -mx-2 rounded-lg bg-amber-50 text-amber-900">
                <span className="shrink-0 w-5 h-5 rounded-lg bg-amber-100 text-amber-700 font-black flex items-center justify-center tabular-nums">
                  3
                </span>
                <span className="font-bold">
                  กรอกราคาให้ {result.missingPrice.toLocaleString()} แถวที่ยังว่าง ใน POS
                </span>
              </li>
            )}
          </ol>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => triggerDownload(result.blob, result.fileName)}
            className={`flex items-center gap-2 px-3 py-2 text-sm font-bold text-slate-600 bg-slate-100 rounded-lg hover:bg-slate-200 ${FOCUS_RING}`}
            data-testid="pos-export-redownload"
          >
            <Download className="w-4 h-4" />
            ดาวน์โหลดไฟล์นี้อีกครั้ง
          </button>
          <button
            onClick={() => {
              setResult(null)
              void fetchPreview()
            }}
            className={`flex items-center gap-2 px-3 py-2 text-sm font-bold text-slate-600 bg-slate-100 rounded-lg hover:bg-slate-200 ${FOCUS_RING}`}
            data-testid="pos-export-dismiss"
          >
            <RefreshCcw className="w-4 h-4" />
            กลับไปหน้ารายการ
          </button>
        </div>

        <p className="text-xs text-slate-400">
          การ์ดนี้ค้างอยู่จนกดโหลดใหม่ หรือปิดแท็บ — ไม่หายเองใน 3 วินาทีแบบ toast
        </p>
      </div>
    )
  }

  const total = data?.total ?? 0
  const issues = data?.issues ?? NO_ISSUES
  const preview = data?.preview ?? []
  const gaps = MANDATORY_FIELDS.filter(({ key }) => issues[key] > 0)
  const readyCount = MANDATORY_FIELDS.length - gaps.length

  const header = (
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div className="min-w-0">
        <h3 className="text-lg font-black text-slate-800">ส่งออกเข้า POS</h3>
        <p className="text-sm text-slate-500">
          สินค้าใหม่ที่ยังไม่มีในสตอก และยังไม่เคยส่งออก
        </p>
      </div>
      <button
        onClick={() => void fetchPreview()}
        className={`flex items-center gap-2 px-3 py-2 text-sm font-bold text-slate-600 bg-slate-100 rounded-lg hover:bg-slate-200 ${FOCUS_RING}`}
        data-testid="pos-export-refresh"
      >
        <RefreshCcw className="w-4 h-4" />
        โหลดใหม่
      </button>
    </div>
  )

  const errorBox = error && (
    <div className="flex items-center gap-2 p-3 text-sm font-bold text-rose-700 bg-rose-50 border border-rose-200 rounded-lg">
      <AlertCircle className="w-4 h-4 shrink-0" />
      {error}
    </div>
  )

  // ---------- State 2: ไม่มีอะไรค้าง ----------
  if (total === 0) {
    return (
      <div data-testid="pos-export-tab" className="thai-text space-y-4">
        {header}
        {errorBox}
        <div
          data-testid="pos-export-empty"
          className="flex flex-col items-center gap-3 py-14 px-4 bg-white border border-slate-200 rounded-xl text-center"
        >
          <CheckCircle2 className="w-9 h-9 text-emerald-500" />
          <p className="font-black text-slate-700">ส่งออกครบแล้ว ไม่มีรายการค้าง</p>
          <p className="text-xs text-slate-400">มีของใหม่เมื่อไหร่จะขึ้นที่นี่เอง</p>
          <button
            data-testid="pos-export-download"
            disabled
            className="flex items-center gap-2 px-4 py-2 mt-1 text-sm font-bold rounded-lg bg-slate-200 text-slate-500 cursor-not-allowed"
          >
            <Download className="w-4 h-4" />
            ดาวน์โหลด .txt
          </button>
        </div>
      </div>
    )
  }

  // ---------- State 3 / 4: มีของรอส่งออก ----------
  const motionProps = {
    initial: { opacity: 0, y: 6 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: prefersReducedMotion ? 0 : 0.2 },
  }

  return (
    <div data-testid="pos-export-tab" className="thai-text space-y-4">
      {header}
      {errorBox}

      <AnimatePresence initial={false}>
        {gaps.length > 0 && (
          <motion.div
            key="warning"
            layout
            {...motionProps}
            data-testid="pos-export-warning"
            className="flex gap-3 p-4 bg-amber-50 border border-amber-200 rounded-xl text-sm text-amber-800"
          >
            <AlertTriangle className="w-5 h-5 shrink-0 text-amber-600" />
            <div className="min-w-0 space-y-1">
              <p className="font-black text-amber-900">POS บังคับช่องเหล่านี้ แต่ยังว่างอยู่</p>
              {gaps.map(({ key, label }) => (
                <p key={key} className="break-words">
                  ขาด<strong className="font-black">{label}</strong>{' '}
                  <span className="tabular-nums">{issues[key].toLocaleString()}</span> แถว
                </p>
              ))}
              <p className="text-xs text-amber-700">
                เติมได้ตอนนำเข้ารอบหน้า โดยเลือกคอลัมน์ที่ตรงกันในขั้นจับคู่คอลัมน์
                — ดาวน์โหลดตอนนี้ได้ แต่ POS อาจไม่รับไฟล์
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* ยอดรวม */}
        <div className="p-5 bg-white border border-slate-200 rounded-xl" data-testid="pos-export-total">
          <p className="text-sm font-bold text-slate-500">รอส่งออกเข้า POS</p>
          <p className="flex items-baseline gap-2 mt-1">
            <span className="text-5xl font-black text-slate-800 tabular-nums">
              {total.toLocaleString()}
            </span>
            <span className="text-sm font-bold text-slate-500">รายการ</span>
          </p>
          <hr className="my-4 border-slate-200" />
          <p className="text-xs text-slate-400">
            คัดจาก 3,103 รายการในสตอก ผ่านการตรวจ 3 แท็บก่อนหน้าแล้ว
          </p>
        </div>

        {/* ความพร้อมของไฟล์ */}
        <div className="p-5 bg-white border border-slate-200 rounded-xl">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h4 className="text-sm font-black text-slate-800">
              ความพร้อมของไฟล์ — ช่องบังคับ 6 ช่อง
            </h4>
            <span
              className={`px-2 py-1 text-xs font-black rounded-lg tabular-nums ${
                gaps.length > 0
                  ? 'bg-amber-100 text-amber-800'
                  : 'bg-emerald-100 text-emerald-700'
              }`}
              data-testid="pos-export-readiness"
            >
              {gaps.length > 0 ? `ครบ ${readyCount} จาก 6` : 'ครบทั้ง 6 ช่อง'}
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5 mt-3">
            {MANDATORY_FIELDS.map(({ key, label }) =>
              issues[key] > 0 ? (
                <div
                  key={key}
                  className="flex items-center gap-1.5 px-2 py-1.5 rounded-lg bg-amber-50 text-amber-900 min-w-0"
                >
                  <AlertTriangle className="w-4 h-4 shrink-0 text-amber-600" />
                  <span className="text-xs font-bold truncate">{label}</span>
                  <span className="ml-auto shrink-0 text-xs font-black text-amber-800 tabular-nums">
                    ขาด {issues[key].toLocaleString()} แถว
                  </span>
                </div>
              ) : (
                <div
                  key={key}
                  className="flex items-center gap-1.5 px-2 py-1.5 rounded-lg bg-emerald-50 text-emerald-700 min-w-0"
                >
                  <Check className="w-4 h-4 shrink-0 text-emerald-600" />
                  <span className="text-xs font-bold truncate">{label}</span>
                  <span className="ml-auto shrink-0 text-xs font-black tabular-nums">
                    {total.toLocaleString()}/{total.toLocaleString()}
                  </span>
                </div>
              )
            )}
          </div>
        </div>
      </div>

      {/* ผลของการกดปุ่ม + ปุ่มหลัก */}
      <div className="flex flex-col lg:flex-row lg:items-center gap-4 p-5 bg-white border border-slate-200 rounded-xl">
        <div className="min-w-0 flex-1">
          <h4 className="text-sm font-black text-slate-800 mb-2">
            กดปุ่มนี้แล้วเกิด {gaps.length > 0 ? 3 : 2} อย่างพร้อมกัน
          </h4>
          <ol className="space-y-1.5 text-xs text-slate-600">
            <li className="flex gap-2">
              <span className="shrink-0 w-5 h-5 rounded-lg bg-slate-100 text-slate-600 font-black flex items-center justify-center tabular-nums">
                1
              </span>
              <span>
                ออก<strong className="font-black">รหัสสินค้าถาวร</strong>ให้ทั้ง{' '}
                <span className="tabular-nums">{total.toLocaleString()}</span> รายการ — แก้คืนไม่ได้
              </span>
            </li>
            <li className="flex gap-2">
              <span className="shrink-0 w-5 h-5 rounded-lg bg-slate-100 text-slate-600 font-black flex items-center justify-center tabular-nums">
                2
              </span>
              <span>ล้างรายการทั้งหมดออกจากคิวหน้านี้ (ตีว่าส่งออกแล้ว)</span>
            </li>
            {gaps.length > 0 && (
              <li className="flex gap-2 p-2 -mx-2 rounded-lg bg-amber-50 text-amber-900">
                <span className="shrink-0 w-5 h-5 rounded-lg bg-amber-100 text-amber-700 font-black flex items-center justify-center tabular-nums">
                  3
                </span>
                <span className="font-bold">
                  ได้ไฟล์ที่ยังไม่มี{gaps.map((gap) => gap.label).join(' / ')} — POS อาจไม่รับ
                </span>
              </li>
            )}
          </ol>
        </div>

        <div className="shrink-0 flex flex-col items-stretch lg:items-end gap-1.5">
          <button
            ref={downloadButtonRef}
            data-testid="pos-export-download"
            onClick={handleDownloadClick}
            disabled={downloading}
            className={`flex items-center justify-center gap-2 px-4 py-2.5 text-sm font-bold rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 disabled:bg-slate-200 disabled:text-slate-500 disabled:cursor-not-allowed ${FOCUS_RING}`}
          >
            {downloading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Download className="w-4 h-4" />
            )}
            {downloading ? 'กำลังสร้างไฟล์...' : 'ดาวน์โหลด .txt'}
          </button>
          <p className="text-xs text-slate-400 text-center lg:text-right">
            มีหน้ายืนยันอีกชั้นก่อนเขียนจริง
          </p>
        </div>
      </div>

      {/* ตัวอย่างข้อมูลในไฟล์ */}
      <div className="p-5 bg-white border border-slate-200 rounded-xl">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
          <h4 className="text-sm font-black text-slate-800">ตัวอย่างข้อมูลในไฟล์</h4>
          <span className="px-2 py-1 text-xs font-bold rounded-lg bg-slate-100 text-slate-500 tabular-nums">
            เห็น {Math.min(preview.length, total).toLocaleString()} แถวแรก · ไฟล์จริง{' '}
            {total.toLocaleString()} แถว
          </span>
        </div>

        {/* จอกว้าง: ตารางเต็ม 9 คอลัมน์ เลื่อนแนวนอนในกรอบของตัวเองเท่านั้น */}
        <div className="hidden md:block overflow-x-auto border border-slate-200 rounded-lg">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 text-xs font-black uppercase text-slate-400">
              <tr>
                <th className="whitespace-nowrap px-3 py-2.5">หมวดสินค้า</th>
                <th className="whitespace-nowrap px-3 py-2.5">หมวดย่อย 1</th>
                <th className="whitespace-nowrap px-3 py-2.5">หมวดย่อย 2</th>
                <th className="whitespace-nowrap px-3 py-2.5">รหัส</th>
                <th className="whitespace-nowrap px-3 py-2.5">บาร์โค้ด</th>
                <th className="whitespace-nowrap px-3 py-2.5">ชื่อสินค้า</th>
                <th className="whitespace-nowrap px-3 py-2.5">หน่วย</th>
                <th className="whitespace-nowrap px-3 py-2.5">ราคา</th>
                <th className="whitespace-nowrap px-3 py-2.5">VAT</th>
              </tr>
            </thead>
            <tbody>
              {preview.map((row, index) => (
                <tr
                  key={`${row.itemId}-${row.itemName}-${index}`}
                  data-testid={`pos-export-row-${index}`}
                  className="border-t border-slate-100"
                >
                  <td className="px-3 py-2.5 text-slate-500 whitespace-nowrap">{row.itemGroup}</td>
                  <td className="px-3 py-2.5 text-slate-500 whitespace-nowrap">{row.subGroup1}</td>
                  <td className="px-3 py-2.5 text-slate-400 whitespace-nowrap">
                    {row.subGroup2 || '—'}
                  </td>
                  <td className="px-3 py-2.5 font-bold text-slate-700 tabular-nums whitespace-nowrap">
                    {row.itemId || '—'}
                  </td>
                  <td className="px-3 py-2.5 text-slate-500 tabular-nums whitespace-nowrap">
                    {row.barcode || '—'}
                  </td>
                  <td className="px-3 py-2.5 font-bold text-slate-700 max-w-[260px] truncate">
                    {row.itemName}
                  </td>
                  <td className="px-3 py-2.5 text-slate-500 whitespace-nowrap">
                    {row.unitName || '—'}
                  </td>
                  <td className="px-3 py-2.5 whitespace-nowrap">
                    {row.price === null ? (
                      <span className="px-2 py-0.5 text-xs font-black rounded-lg bg-amber-100 text-amber-800">
                        ว่าง
                      </span>
                    ) : (
                      <span className="text-slate-700 font-bold tabular-nums">
                        {row.price.toLocaleString()}
                      </span>
                    )}
                  </td>
                  <td className="px-3 py-2.5 text-slate-500 whitespace-nowrap">{row.vat}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* จอแคบ: การ์ดต่อสินค้า ไม่มีอะไรล้นแนวนอนเลย */}
        <div className="md:hidden space-y-2">
          {preview.map((row, index) => (
            <div
              key={`${row.itemId}-${row.itemName}-${index}`}
              data-testid={`pos-export-card-${index}`}
              className="p-3 border border-slate-200 rounded-lg min-w-0"
            >
              <div className="flex items-center justify-between gap-2 mb-1">
                <span className="text-xs font-black text-slate-700 tabular-nums truncate">
                  {row.itemId || '—'}
                </span>
                {row.price === null ? (
                  <span className="shrink-0 px-2 py-1 text-xs font-black rounded-lg bg-amber-100 text-amber-800">
                    ราคาว่าง
                  </span>
                ) : (
                  <span className="shrink-0 px-2 py-1 text-xs font-black rounded-lg bg-emerald-100 text-emerald-700 tabular-nums">
                    ครบ · {row.price.toLocaleString()}
                  </span>
                )}
              </div>
              <p className="text-xs font-bold text-slate-800 break-words">{row.itemName}</p>
              <p className="text-xs text-slate-400 break-words">
                {[row.itemGroup, row.subGroup1, row.unitName, `VAT ${row.vat}`]
                  .filter(Boolean)
                  .join(' · ')}
              </p>
            </div>
          ))}
        </div>

        <p className="mt-2 text-xs text-slate-400">
          ตารางเลื่อนแนวนอนในกรอบของตัวเอง หน้าไม่เลื่อนตาม
        </p>
      </div>

      {/* กล่องยืนยัน — ขั้นสุดท้ายก่อนเขียนฐานข้อมูลถาวร */}
      {confirmOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/55"
          onClick={closeConfirm}
        >
          <div
            ref={dialogRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby="pos-export-confirm-title"
            data-testid="pos-export-confirm-dialog"
            onClick={(event) => event.stopPropagation()}
            className="w-full max-w-[470px] max-h-full overflow-y-auto p-6 bg-white rounded-xl shadow-2xl thai-text"
          >
            <div className="flex items-start gap-3">
              <span className="shrink-0 flex items-center justify-center w-10 h-10 rounded-lg bg-indigo-50">
                <FileText className="w-5 h-5 text-indigo-600" />
              </span>
              <div className="min-w-0">
                <h4 id="pos-export-confirm-title" className="text-base font-black text-slate-800">
                  ยืนยันส่งออก {total.toLocaleString()} รายการ
                </h4>
                <p className="text-xs text-slate-500">ตรวจ 3 ข้อนี้ก่อนกด</p>
              </div>
            </div>

            <div className="mt-4 p-3 bg-slate-50 border border-slate-200 rounded-lg space-y-1.5 text-xs">
              <p className="flex items-center justify-between gap-2">
                <span className="text-slate-600">ออกรหัสสินค้าถาวร</span>
                <span className="font-black text-slate-800 tabular-nums">
                  {total.toLocaleString()} รหัส
                </span>
              </p>
              <p className="flex items-center justify-between gap-2">
                <span className="text-slate-600">ล้างออกจากคิวหน้านี้</span>
                <span className="font-black text-slate-800 tabular-nums">
                  {total.toLocaleString()} รายการ
                </span>
              </p>
              {gaps.map(({ key, label }) => (
                <p key={key} className="flex items-center justify-between gap-2">
                  <span className="text-slate-600">ช่อง{label}ที่ยังว่าง</span>
                  <span className="font-black text-amber-700 tabular-nums">
                    {issues[key].toLocaleString()} แถว
                  </span>
                </p>
              ))}
            </div>

            <label className="flex items-center gap-2 mt-4 text-xs font-bold text-slate-600 cursor-pointer">
              <input
                type="checkbox"
                checked={dontAskAgain}
                onChange={(event) => setDontAskAgain(event.target.checked)}
                className={`w-4 h-4 rounded border-slate-300 text-indigo-600 ${FOCUS_RING}`}
                data-testid="pos-export-skip-confirm"
              />
              ไม่ต้องถามอีกในเซสชันนี้
            </label>

            <div className="flex flex-wrap justify-end gap-2 mt-5">
              <button
                onClick={closeConfirm}
                className={`px-4 py-2 text-sm font-bold rounded-lg bg-slate-100 text-slate-600 hover:bg-slate-200 ${FOCUS_RING}`}
                data-testid="pos-export-cancel"
              >
                ยกเลิก
              </button>
              <button
                autoFocus
                onClick={() => void handleConfirm()}
                className={`flex items-center gap-2 px-4 py-2 text-sm font-bold rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 ${FOCUS_RING}`}
                data-testid="pos-export-confirm"
              >
                <Hash className="w-4 h-4" />
                ยืนยัน ดาวน์โหลดไฟล์
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
