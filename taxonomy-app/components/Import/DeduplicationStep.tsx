'use client'

import { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  CopyIcon, CheckCircleIcon, ArrowRightIcon, AlertTriangleIcon,
  ScaleIcon, WifiOffIcon, RefreshCwIcon
} from 'lucide-react'
import { isPriceMismatch, classifyDedupBucket } from '@/utils/price'
import type { DedupBucket, DedupResults, WizardItem } from '@/types/import'

type DedupItem = WizardItem

interface DeduplicationStepProps {
  cleanedData: WizardItem[]
  onComplete: (deduped: WizardItem[]) => void
  onBack?: () => void
}

export default function DeduplicationStep({
  cleanedData,
  onComplete,
  onBack
}: DeduplicationStepProps) {
  const [isProcessing, setIsProcessing] = useState(true)
  const [progress, setProgress] = useState(0)
  const [statusMsg, setStatusMsg] = useState('กำลังเชื่อมต่อระบบแนะนำอัจฉริยะ...')
  const [isMock, setIsMock] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dedupResults, setDedupResults] = useState<DedupResults>({ 
    autoMerged: [], autoCreated: [], reviewZone: [] 
  })
  // ตรวจทีละรายการเหมือนหน้า Data Quality → ตรวจของซ้ำในคลัง
  // เก็บผลไว้ในเครื่องก่อน ยิงขึ้นฐานข้อมูลทีเดียวตอนจบขั้นตอน — ต่างจากหน้านั้นตรงที่
  // สินค้ายังไม่มีอยู่จริงในฐานข้อมูล เพิ่งจะถูกสร้างตอน commit จึงยิง API รายตัวไม่ได้
  const [reviewIndex, setReviewIndex] = useState(0)

  useEffect(() => {
    runDeduplication()
  }, [cleanedData])

  const runDeduplication = async () => {
    setIsProcessing(true)
    setProgress(10)
    setStatusMsg('กำลังส่งข้อมูลไปยังระบบประมวลผลอัจฉริยะ...')
    setError(null)

    try {
      const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'

      setProgress(30)
      setStatusMsg(`กำลังส่งสินค้าใหม่ให้ AI ทำการเปรียบเทียบเชิงลึก...`)

      // Extract all cleaned names to send to FastAPI
      const productNames = cleanedData
        .map((item: any) => item._cleaned_name || item.name_th || item.name || '')
        .filter(Boolean)

      const res = await fetch(`${apiBase}/api/v1/match/import-dedup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ products: productNames, threshold: 0.75 }),
        signal: AbortSignal.timeout(120000)
      })

      setProgress(70)
      setStatusMsg('กำลังวิเคราะห์และจัดหมวดหมู่ความซ้ำซ้อนด้วย Machine Learning...')

      if (!res.ok) {
        throw new Error(`Backend ตอบกลับ ${res.status}: ${await res.text()}`)
      }

      const dbPairs: any[] = await res.json()
      setProgress(85)

      // Create lookup map from query name to matched result
      const matchMap = new Map<string, any>()
      dbPairs.forEach((match: any) => {
        matchMap.set(match.newProduct?.toLowerCase(), match)
      })

      const autoMerged: DedupItem[] = []
      const autoCreated: DedupItem[] = []
      const reviewZone: DedupItem[] = []

      cleanedData.forEach((item: any) => {
        const itemName = (item._cleaned_name || item.name_th || item.name || '').toLowerCase()
        const match = matchMap.get(itemName)

        if (match) {
          const score = match.similarity || 0
          const priceMismatch = isPriceMismatch(item.price, match.oldPrice)
          const enriched: DedupItem = {
            ...item,
            _similarity_score: score,
            _matched_with: match.oldProduct || 'สินค้าในระบบ',
            // ต้องใช้ oldProductId (id จริงของสินค้าในคลัง) — `match.id` เป็นเลขลำดับรีวิว
            // เช่น "review_1" ใช้เป็น FK ไม่ได้
            _matched_id: match.oldProductId,
            _source: 'backend',
            // ราคาแนบไว้เสมอเมื่อมี match — ใช้โชว์เป็นข้อมูลช่วยตรวจใน review zone
            // ไม่ว่าจะ mismatch หรือไม่ (ราคาตรงกันก็ช่วยยืนยันว่าน่าจะเป็นของซ้ำจริง)
            _new_price: item.price,
            _old_price: match.oldPrice,
            _price_mismatch: priceMismatch
          }

          // _bucket ต้องติดไปกับข้อมูลด้วย ขั้นถัดไปใช้ตัดสินว่าจะบันทึกด้วยสถานะอะไร
          const bucket = classifyDedupBucket(score, match.mlPrediction, item.price, match.oldPrice)
          if (bucket === 'duplicate') {
            autoMerged.push({ ...enriched, _bucket: 'duplicate' })
          } else if (bucket === 'review') {
            reviewZone.push({ ...enriched, _bucket: 'review' })
          } else {
            autoCreated.push({ ...enriched, _bucket: 'new' })
          }
        } else {
          autoCreated.push({
            ...item,
            _similarity_score: 0,
            _matched_with: '',
            _bucket: 'new',
            _source: 'backend'
          })
        }
      })

      setDedupResults({ autoMerged, autoCreated, reviewZone })
      // ตรวจใหม่ทั้งชุด ต้องกลับไปเริ่มที่รายการแรกเสมอ
      setReviewIndex(0)
      setIsMock(false)
      setProgress(100)
    } catch (err: any) {
      console.warn('Backend dedup failed, using mock:', err.message)
      setError(err.message)
      setStatusMsg('Backend ไม่ตอบสนอง — ใช้ข้อมูลจำลองแทน')
      setIsMock(true)
      // Fallback mock
      fallbackMock()
    } finally {
      setTimeout(() => setIsProcessing(false), 300)
    }
  }

  const fallbackMock = () => {
    const autoMerged: DedupItem[] = []
    const autoCreated: DedupItem[] = []
    const reviewZone: DedupItem[] = []

    cleanedData.forEach((item: any, index: number) => {
      const mockScore = 0.6 + (Math.random() * 0.4)
      const enriched: DedupItem = {
        ...item,
        _similarity_score: mockScore,
        _matched_with: `สินค้าอ้างอิง_${Math.floor(Math.random() * 100)}`,
        _source: 'mock'
      }
      if (mockScore >= 0.95) autoMerged.push({ ...enriched, _bucket: 'duplicate' })
      else if (mockScore <= 0.79) autoCreated.push({ ...enriched, _bucket: 'new' })
      else reviewZone.push({ ...enriched, _bucket: 'review' })
    })
    setDedupResults({ autoMerged, autoCreated, reviewZone })
    setReviewIndex(0)
    setProgress(100)
  }

  /**
   * บันทึกผลตัดสินของรายการปัจจุบันแล้วเลื่อนไปรายการถัดไป
   *
   * 'review' = ยังไม่ตัดสิน ปล่อยให้ไปตัดสินต่อที่หน้า Data Quality → Verify
   * ซึ่งเป็นพฤติกรรมเดิมของทุกรายการในโซนนี้ การกดข้ามจึงไม่ทำให้เสียข้อมูล
   */
  const decideCurrent = useCallback((bucket: DedupBucket) => {
    setDedupResults((prev) => {
      if (!prev.reviewZone[reviewIndex]) return prev
      const reviewZone = prev.reviewZone.map((item, i) =>
        i === reviewIndex ? { ...item, _bucket: bucket, _reviewed_by_user: true } : item
      )
      return { ...prev, reviewZone }
    })
    setReviewIndex((i) => Math.min(i + 1, dedupResults.reviewZone.length))
  }, [reviewIndex, dedupResults.reviewZone.length])

  const currentReviewItem = dedupResults.reviewZone[reviewIndex]
  // ต้องระบุชนิดของตัวสะสมเอง — WizardItem มี index signature จึงรับ object literal นี้ได้ด้วย
  // TypeScript เลยเลือก overload ที่ตัวสะสมเป็น WizardItem แล้วค่าที่นับออกมากลายเป็น unknown
  const reviewDecidedCounts = dedupResults.reviewZone.reduce<{
    duplicate: number
    new: number
    skipped: number
  }>(
    (acc, item) => {
      if (!item._reviewed_by_user) return acc
      if (item._bucket === 'duplicate') acc.duplicate += 1
      else if (item._bucket === 'new') acc.new += 1
      else acc.skipped += 1
      return acc
    },
    { duplicate: 0, new: 0, skipped: 0 }
  )
  const reviewDecidedCount =
    reviewDecidedCounts.duplicate + reviewDecidedCounts.new + reviewDecidedCounts.skipped

  // ปุ่มลัดชุดเดียวกับหน้าตรวจของซ้ำในคลัง รวมทั้งผังแป้นไทย
  // ผู้ใช้ที่ชินกับหน้านั้นแล้วจะใช้ที่นี่ได้ทันทีโดยไม่ต้องจำใหม่
  useEffect(() => {
    if (isProcessing || !dedupResults.reviewZone[reviewIndex]) return

    const handleKeyDown = (e: KeyboardEvent) => {
      const tag = document.activeElement?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA') return

      const key = e.key.toLowerCase()
      const code = e.code

      if (code === 'KeyA' || code === 'ArrowLeft' || key === 'a' || key === 'ฟ' || key === 'ฤ') {
        e.preventDefault()
        decideCurrent('duplicate')
      } else if (code === 'KeyD' || code === 'ArrowRight' || key === 'd' || key === 'ก' || key === 'ฏ') {
        e.preventDefault()
        decideCurrent('new')
      } else if (code === 'KeyS' || code === 'ArrowDown' || key === 's' || key === 'ห' || key === 'ฆ') {
        e.preventDefault()
        decideCurrent('review')
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isProcessing, dedupResults.reviewZone, reviewIndex, decideCurrent])

  if (isProcessing) {
    return (
      <div className="flex flex-col items-center justify-center py-32 space-y-6">
        <div className="relative">
          <div className="w-24 h-24 border-4 border-indigo-100 border-t-indigo-600 rounded-full animate-spin"></div>
          <CopyIcon className="w-8 h-8 text-indigo-600 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-pulse" />
        </div>
        <div className="text-center space-y-3 w-80">
          <h3 className="text-xl font-bold text-slate-800 thai-text">กำลังค้นหาสินค้าซ้ำด้วยระบบประมวลผลอัจฉริยะ...</h3>
          <p className="text-slate-500 thai-text text-sm">{statusMsg}</p>
          <div className="w-full bg-slate-100 rounded-full h-2">
            <motion.div 
              className="h-2 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
          <p className="text-xs text-slate-400">{progress}%</p>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto space-y-8 pb-20">
      {/* Source Badge */}
      {isMock && (
        <div className="bg-amber-50 border border-amber-200 rounded-2xl p-4 flex items-center gap-3">
          <WifiOffIcon className="w-5 h-5 text-amber-600 shrink-0" />
          <div className="flex-1">
            <p className="font-bold text-amber-800 thai-text text-sm">ระบบประมวลผลอัจฉริยะไม่พร้อมใช้งาน</p>
            <p className="text-amber-600 text-xs thai-text">{error} — แสดงผลจำลองเท่านั้น</p>
          </div>
          <button 
            onClick={runDeduplication}
            className="flex items-center gap-1 text-amber-700 text-sm font-bold hover:text-amber-900 transition-colors"
          >
            <RefreshCwIcon className="w-4 h-4" /> ลองใหม่
          </button>
        </div>
      )}

      {/* Header Info */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-black text-slate-900 tracking-tight font-noto-sans-thai mb-2 flex items-center gap-3">
            <CopyIcon className="w-8 h-8 text-indigo-600" />
            จัดการสินค้าชื่อคล้าย (Deduplication)
          </h2>
          <p className="text-slate-500 font-medium thai-text">
            ระบบได้จัดกลุ่มสินค้าตามความคล้ายคลึงเชิงความหมาย{!isMock && ' — ข้อมูลจากระบบวิเคราะห์จริง'} เพื่อลดภาระการตรวจสอบของคุณ
          </p>
        </div>
      </div>

      {/* Triage Dashboard */}
      <div className="grid grid-cols-3 gap-6">
        <div className="bg-emerald-50 border border-emerald-100 rounded-3xl p-6 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-32 h-32 bg-emerald-500/10 rounded-full blur-2xl -mr-16 -mt-16" />
          <h4 className="text-emerald-800 font-bold thai-text mb-1">รวมเป็นของชิ้นเดียวกันอัตโนมัติ</h4>
          <p className="text-emerald-600/70 text-xs font-bold uppercase tracking-wider mb-4">มั่นใจ &gt; 95%</p>
          <div className="text-4xl font-black text-emerald-600">{dedupResults.autoMerged.length}</div>
        </div>
        <div className="bg-amber-50 border border-amber-100 rounded-3xl p-6 relative overflow-hidden ring-4 ring-amber-500/20">
          <div className="absolute top-0 right-0 w-32 h-32 bg-amber-500/10 rounded-full blur-2xl -mr-16 -mt-16" />
          <h4 className="text-amber-800 font-bold thai-text mb-1 flex items-center gap-2">
            รอการตรวจสอบ <AlertTriangleIcon className="w-4 h-4" />
          </h4>
          <p className="text-amber-600/70 text-xs font-bold uppercase tracking-wider mb-4">ก้ำกึ่ง 80-94%</p>
          <div className="text-4xl font-black text-amber-600">{dedupResults.reviewZone.length}</div>
        </div>
        <div className="bg-blue-50 border border-blue-100 rounded-3xl p-6 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/10 rounded-full blur-2xl -mr-16 -mt-16" />
          <h4 className="text-blue-800 font-bold thai-text mb-1">สร้างเป็นของใหม่</h4>
          <p className="text-blue-600/70 text-xs font-bold uppercase tracking-wider mb-4">ความเหมือน &lt; 80%</p>
          <div className="text-4xl font-black text-blue-600">{dedupResults.autoCreated.length}</div>
        </div>
      </div>

      {/* Review Zone — ตรวจทีละรายการ */}
      <div className="premium-card bg-white border border-slate-200 shadow-xl rounded-3xl overflow-hidden">
        <div className="p-6 border-b border-slate-100 bg-slate-50/50 flex items-center justify-between">
          <h3 className="font-bold text-slate-800 thai-text">
            รายการที่ต้องตรวจสอบ ({dedupResults.reviewZone.length})
          </h3>
          {dedupResults.reviewZone.length > 0 && (
            <span className="px-5 py-2 bg-indigo-500 rounded-2xl text-white font-black text-sm shadow-lg shadow-indigo-100">
              {Math.min(reviewIndex + 1, dedupResults.reviewZone.length)} / {dedupResults.reviewZone.length}
            </span>
          )}
        </div>

        {currentReviewItem ? (
          <div className="p-8 space-y-8">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 relative">
              <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-20 hidden lg:flex w-16 h-16 rounded-full bg-white shadow-2xl border border-slate-100 items-center justify-center">
                <div className="text-xs font-black text-amber-500">
                  {((currentReviewItem._similarity_score ?? 0) * 100).toFixed(0)}%
                </div>
              </div>

              <div className="border border-slate-100 rounded-3xl overflow-hidden">
                <div className="px-6 py-3 bg-indigo-600 text-white">
                  <span className="text-xs font-black uppercase tracking-[0.2em] thai-text">สินค้าที่อัปโหลด</span>
                </div>
                <div className="p-8 min-h-[140px] flex flex-col justify-between">
                  <p className="text-lg font-black text-slate-800 thai-text leading-relaxed">
                    {currentReviewItem._cleaned_name}
                  </p>
                  {currentReviewItem._new_price != null && (
                    <p className="text-sm font-bold text-slate-500 mt-4">฿{currentReviewItem._new_price}</p>
                  )}
                </div>
              </div>

              <div className="border border-slate-100 rounded-3xl overflow-hidden">
                <div className="px-6 py-3 bg-emerald-600 text-white">
                  <span className="text-xs font-black uppercase tracking-[0.2em] thai-text">สินค้าในระบบ (อาจจะซ้ำ)</span>
                </div>
                <div className="p-8 min-h-[140px] flex flex-col justify-between">
                  <p className="text-lg font-black text-slate-700 thai-text leading-relaxed">
                    {currentReviewItem._matched_with}
                  </p>
                  {currentReviewItem._old_price != null && (
                    <p className="text-sm font-bold text-slate-500 mt-4">฿{currentReviewItem._old_price}</p>
                  )}
                </div>
              </div>
            </div>

            {currentReviewItem._price_mismatch && (
              <div className="flex items-center gap-2 text-xs font-bold text-rose-600 bg-rose-50 px-4 py-2.5 rounded-xl w-fit mx-auto thai-text">
                <AlertTriangleIcon className="w-4 h-4" />
                ราคาต่างกันมาก (฿{currentReviewItem._old_price} → ฿{currentReviewItem._new_price}) อาจเป็นคนละสินค้า
              </div>
            )}

            <div className="bg-slate-950 p-8 rounded-[2.5rem] text-center relative overflow-hidden">
              <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-500/10 rounded-full blur-[80px] -mr-32 -mt-32" />
              <div className="relative z-10">
                <h4 className="text-base font-black text-white thai-text mb-2">สินค้าชิ้นนี้ซ้ำกับของในสตอกหรือไม่</h4>
                <p className="text-[10px] text-slate-400 font-bold uppercase tracking-widest mb-8 thai-text">
                  ใช้ปุ่มลัด [A] [S] [D] หรือปุ่มลูกศร [←] [↓] [→] เพื่อทำงานได้เร็วขึ้น
                </p>

                <div className="flex flex-col md:flex-row justify-center items-stretch gap-4">
                  <button
                    onClick={() => decideCurrent('duplicate')}
                    className="flex-1 px-6 py-4 bg-emerald-500 hover:bg-emerald-400 text-white rounded-2xl font-black text-sm transition-all active:scale-95 thai-text"
                  >
                    ซ้ำ — มีอยู่แล้วในสตอก
                    <span className="block text-[10px] font-bold text-emerald-100 mt-1 tracking-widest">[A] / [←]</span>
                  </button>
                  <button
                    onClick={() => decideCurrent('review')}
                    className="flex-1 px-6 py-4 bg-white/10 hover:bg-white/20 text-white rounded-2xl font-black text-sm transition-all active:scale-95 thai-text"
                  >
                    ยังไม่ตัดสิน — ไว้ดูที่หน้า Verify
                    <span className="block text-[10px] font-bold text-slate-400 mt-1 tracking-widest">[S] / [↓]</span>
                  </button>
                  <button
                    onClick={() => decideCurrent('new')}
                    className="flex-1 px-6 py-4 bg-blue-500 hover:bg-blue-400 text-white rounded-2xl font-black text-sm transition-all active:scale-95 thai-text"
                  >
                    คนละตัว — เป็นของใหม่
                    <span className="block text-[10px] font-bold text-blue-100 mt-1 tracking-widest">[D] / [→]</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="p-12 text-center text-slate-500 thai-text">
            {dedupResults.reviewZone.length === 0
              ? 'ไม่มีรายการที่ต้องตรวจสอบเพิ่มเติม ยอดเยี่ยมมาก!'
              : `ตรวจครบทั้ง ${dedupResults.reviewZone.length} รายการแล้ว — กดดำเนินการต่อได้เลย`}
          </div>
        )}

        {reviewDecidedCount > 0 && (
          <div className="px-6 py-4 border-t border-slate-100 bg-slate-50/50 flex items-center gap-6 text-xs font-bold thai-text">
            <span className="text-emerald-700">ตัดสินว่าซ้ำ {reviewDecidedCounts.duplicate}</span>
            <span className="text-blue-700">ตัดสินว่าของใหม่ {reviewDecidedCounts.new}</span>
            <span className="text-slate-500">
              ยังไม่ตัดสิน {dedupResults.reviewZone.length - reviewDecidedCount} — จะไปรอที่หน้า Data Quality → Verify
            </span>
          </div>
        )}
      </div>

      {/* Footer Actions */}
      <div className="flex items-center justify-between pt-8">
        <button
          onClick={onBack}
          className="px-8 py-4 text-slate-500 hover:text-slate-900 font-bold text-sm transition-all thai-text"
        >
          ← ย้อนกลับ
        </button>

        <button
          onClick={() => onComplete([
            // ส่งผลที่แบ่งกลุ่มแล้วออกไป ไม่ใช่ข้อมูลดิบ
            // เดิมส่ง cleanedData กลับไปเฉยๆ ทำให้ผลตรวจของซ้ำถูกทิ้งทั้งหมด
            // ขั้นถัดไปจึงจัดหมวดสินค้าทุกตัวรวมทั้งตัวที่มีในสตอกอยู่แล้ว
            ...dedupResults.autoMerged,
            ...dedupResults.reviewZone,
            ...dedupResults.autoCreated,
          ])}
          className="px-10 py-4 bg-slate-900 hover:bg-black text-white rounded-[2rem] font-black text-sm tracking-wide shadow-xl shadow-slate-900/20 transition-all active:scale-95 flex items-center gap-3 thai-text"
        >
          ดำเนินการต่อ: จัดหมวดหมู่
          <ArrowRightIcon className="w-5 h-5" />
        </button>
      </div>
    </div>
  )
}
