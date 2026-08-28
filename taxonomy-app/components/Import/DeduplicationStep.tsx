'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  CopyIcon, CheckCircleIcon, ArrowRightIcon, AlertTriangleIcon,
  SearchIcon, CheckSquareIcon, WifiOffIcon, RefreshCwIcon
} from 'lucide-react'
import { isPriceMismatch, classifyDedupBucket } from '@/utils/price'

interface DedupItem {
  _cleaned_name: string
  _similarity_score: number
  _matched_with: string
  _matched_id?: string
  _source: 'backend' | 'mock'
  [key: string]: any
}

interface DedupResults {
  autoMerged: DedupItem[]
  autoCreated: DedupItem[]
  reviewZone: DedupItem[]
}

export default function DeduplicationStep({
  cleanedData,
  onComplete,
  onBack
}: any) {
  const [isProcessing, setIsProcessing] = useState(true)
  const [progress, setProgress] = useState(0)
  const [statusMsg, setStatusMsg] = useState('กำลังเชื่อมต่อระบบแนะนำอัจฉริยะ...')
  const [isMock, setIsMock] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dedupResults, setDedupResults] = useState<DedupResults>({ 
    autoMerged: [], autoCreated: [], reviewZone: [] 
  })
  const [selectedItems, setSelectedItems] = useState<Set<number>>(new Set())

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
    setProgress(100)
  }

  const toggleSelect = (index: number) => {
    const newSet = new Set(selectedItems)
    if (newSet.has(index)) newSet.delete(index)
    else newSet.add(index)
    setSelectedItems(newSet)
  }

  const selectAll = () => {
    if (selectedItems.size === dedupResults.reviewZone.length) {
      setSelectedItems(new Set())
    } else {
      setSelectedItems(new Set(dedupResults.reviewZone.map((_, i) => i)))
    }
  }

  const handleBulkAction = (action: 'merge' | 'create') => {
    setSelectedItems(new Set())
  }

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

      {/* Review Zone List */}
      <div className="premium-card bg-white border border-slate-200 shadow-xl rounded-3xl overflow-hidden">
        <div className="p-6 border-b border-slate-100 bg-slate-50/50 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h3 className="font-bold text-slate-800 thai-text">รายการที่ต้องตรวจสอบ ({dedupResults.reviewZone.length})</h3>
            {selectedItems.size > 0 && (
              <span className="text-xs font-bold bg-amber-100 text-amber-700 px-3 py-1 rounded-full">
                เลือกแล้ว {selectedItems.size} รายการ
              </span>
            )}
          </div>
          
          {selectedItems.size > 0 ? (
            <div className="flex items-center gap-3">
              <button 
                onClick={() => handleBulkAction('merge')}
                className="px-4 py-2 bg-emerald-100 text-emerald-700 hover:bg-emerald-200 rounded-xl font-bold text-sm thai-text transition-colors"
              >
                รวมเป็นของชิ้นเดียวกัน
              </button>
              <button 
                onClick={() => handleBulkAction('create')}
                className="px-4 py-2 bg-blue-100 text-blue-700 hover:bg-blue-200 rounded-xl font-bold text-sm thai-text transition-colors"
              >
                แยกเป็นของใหม่
              </button>
            </div>
          ) : (
            <div className="relative">
              <SearchIcon className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
              <input 
                type="text" 
                placeholder="ค้นหารายการ..." 
                className="pl-9 pr-4 py-2 bg-white border border-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 w-64"
              />
            </div>
          )}
        </div>

        <div className="divide-y divide-slate-100 max-h-[500px] overflow-y-auto">
          {dedupResults.reviewZone.map((item, index) => (
            <div key={index} className={`p-4 flex items-center gap-6 transition-colors ${selectedItems.has(index) ? 'bg-indigo-50/50' : 'hover:bg-slate-50'}`}>
              <button 
                onClick={() => toggleSelect(index)}
                className={`w-5 h-5 rounded flex items-center justify-center border transition-colors shrink-0 ${
                  selectedItems.has(index) ? 'bg-indigo-600 border-indigo-600 text-white' : 'border-slate-300 text-transparent'
                }`}
              >
                <CheckSquareIcon className="w-3.5 h-3.5 fill-current" />
              </button>
              
              <div className="flex-1">
                <div className="grid grid-cols-2 gap-8 items-center">
                  <div>
                    <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1 thai-text">สินค้าที่อัปโหลด</div>
                    <div className="font-bold text-slate-800">{item._cleaned_name}</div>
                    {item._new_price != null && (
                      <div className="text-xs font-bold text-slate-500 mt-1">฿{item._new_price}</div>
                    )}
                  </div>

                  <div className="relative pl-8 border-l border-slate-200">
                    <div className="absolute -left-3 top-1/2 -translate-y-1/2 bg-white p-1 rounded-full border border-slate-200 shadow-sm">
                      <div className="text-[10px] font-black text-amber-500">{(item._similarity_score * 100).toFixed(0)}%</div>
                    </div>
                    <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-1 thai-text">สินค้าในระบบ (อาจจะซ้ำ)</div>
                    <div className="font-bold text-slate-700">{item._matched_with}</div>
                    {item._old_price != null && (
                      <div className="text-xs font-bold text-slate-500 mt-1">฿{item._old_price}</div>
                    )}
                  </div>
                </div>
                {item._price_mismatch && (
                  <div className="mt-2 flex items-center gap-1 text-[11px] font-bold text-rose-600 bg-rose-50 px-3 py-1.5 rounded-lg w-fit">
                    <AlertTriangleIcon className="w-3.5 h-3.5" />
                    ราคาต่างกันมาก (฿{item._old_price} → ฿{item._new_price}) อาจเป็นคนละสินค้า
                  </div>
                )}
              </div>
            </div>
          ))}
          {dedupResults.reviewZone.length === 0 && (
            <div className="p-12 text-center text-slate-500 thai-text">
              ไม่มีรายการที่ต้องตรวจสอบเพิ่มเติม ยอดเยี่ยมมาก!
            </div>
          )}
        </div>
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
