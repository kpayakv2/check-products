'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { TagsIcon, CheckCircleIcon, ArrowRightIcon, AlertTriangleIcon, CheckIcon, XIcon, ChevronDownIcon, ChevronUpIcon, WifiOffIcon, RefreshCwIcon } from 'lucide-react'

interface CategorySuggestion {
  _cleaned_name: string
  _suggested_category: string
  _suggested_category_id?: string
  _confidence: number
  _status: 'pending' | 'approve' | 'reject'
  _source: 'backend' | 'mock'
  [key: string]: any
}

export default function CategorizationStep({
  dedupedData,
  onComplete,
  onBack
}: any) {
  const [isProcessing, setIsProcessing] = useState(true)
  const [progress, setProgress] = useState(0)
  const [statusMsg, setStatusMsg] = useState('กำลังเชื่อมต่อระบบแนะนำอัจฉริยะ...')
  const [isMock, setIsMock] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [suggestions, setSuggestions] = useState<CategorySuggestion[]>([])
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set())
  const [expandedIds, setExpandedIds] = useState<Set<number>>(new Set())

  useEffect(() => {
    runCategorization()
  }, [dedupedData])

  const runCategorization = async () => {
    setIsProcessing(true)
    setProgress(5)
    setStatusMsg('กำลังเตรียมข้อมูลสินค้า...')
    setError(null)

    try {
      const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'
      const items = dedupedData.filter((item: any) => item._cleaned_name || item.name_th || item.name)

      setProgress(15)
      setStatusMsg(`กำลังวิเคราะห์หมวดหมู่สำหรับสินค้า ${items.length} รายการ...`)

      // Batch classify — ส่งทีละ 10 รายการ เพื่อไม่ให้ request หนักเกินไป
      const BATCH_SIZE = 10
      const results: CategorySuggestion[] = []
      
      for (let i = 0; i < items.length; i += BATCH_SIZE) {
        const batch = items.slice(i, i + BATCH_SIZE)
        
        // ส่ง parallel requests สำหรับ batch นี้
        const batchPromises = batch.map(async (item: any) => {
          const productName = item._cleaned_name || item.name_th || item.name || ''
          try {
            const res = await fetch(`${apiBase}/api/classify/category`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ product_name: productName }),
              signal: AbortSignal.timeout(10000)
            })
            
            if (!res.ok) throw new Error(`${res.status}`)
            const data = await res.json()
            
            return {
              ...item,
              _suggested_category: data.category_name || data.category?.name_th || data.category || 'ไม่ระบุ',
              _suggested_category_id: data.category_id || data.category?.id,
              _confidence: data.confidence || data.score || 0.5,
              _status: 'pending' as const,
              _source: 'backend' as const
            }
          } catch {
            // สินค้าที่ classify ไม่ได้ — ใช้ mock
            return {
              ...item,
              _suggested_category: 'ต้องตรวจสอบ',
              _confidence: 0,
              _status: 'pending' as const,
              _source: 'mock' as const
            }
          }
        })

        const batchResults = await Promise.all(batchPromises)
        results.push(...batchResults)
        
        const pct = Math.round(15 + ((i + batch.length) / items.length) * 80)
        setProgress(pct)
        setStatusMsg(`จัดหมวดหมู่แล้ว ${Math.min(i + BATCH_SIZE, items.length)}/${items.length} รายการ...`)
      }

      setSuggestions(results)
      const mockCount = results.filter(r => r._source === 'mock').length
      if (mockCount > 0 && mockCount === results.length) {
        setIsMock(true)
      }
      setProgress(100)
    } catch (err: any) {
      console.warn('Backend categorization failed:', err.message)
      setError(err.message)
      setIsMock(true)
      setStatusMsg('Backend ไม่ตอบสนอง — ใช้ข้อมูลจำลองแทน')
      fallbackMock()
    } finally {
      setTimeout(() => setIsProcessing(false), 300)
    }
  }

  const fallbackMock = () => {
    const mockCategories = ['เครื่องดื่ม', 'ของใช้ส่วนตัว', 'อาหารแห้ง', 'อุปกรณ์ทำความสะอาด', 'อาหารสด']
    const categorized = dedupedData.map((item: any) => ({
      ...item,
      _suggested_category: mockCategories[Math.floor(Math.random() * mockCategories.length)],
      _confidence: 0.5 + (Math.random() * 0.5),
      _status: 'pending' as const,
      _source: 'mock' as const
    }))
    setSuggestions(categorized)
    setProgress(100)
  }


  const toggleSelection = (index: number) => {
    const newSelected = new Set(selectedIds)
    if (newSelected.has(index)) newSelected.delete(index)
    else newSelected.add(index)
    setSelectedIds(newSelected)
  }

  const toggleSelectAll = () => {
    if (selectedIds.size === suggestions.length) {
      setSelectedIds(new Set())
    } else {
      setSelectedIds(new Set(suggestions.map((_, i) => i)))
    }
  }

  const toggleExpanded = (index: number) => {
    const newExpanded = new Set(expandedIds)
    if (newExpanded.has(index)) newExpanded.delete(index)
    else newExpanded.add(index)
    setExpandedIds(newExpanded)
  }

  const handleBatchAction = (action: 'approve' | 'reject') => {
    setSuggestions(prev => prev.map((s, i) => 
      selectedIds.has(i) ? { ...s, _status: action } : s
    ))
    setSelectedIds(new Set())
  }

  const handleComplete = () => {
    onComplete(suggestions)
  }

  if (isProcessing) {
    return (
      <div className="flex flex-col items-center justify-center py-32 space-y-6">
        <div className="relative">
          <div className="w-24 h-24 border-4 border-indigo-100 border-t-indigo-600 rounded-full animate-spin"></div>
          <TagsIcon className="w-8 h-8 text-indigo-600 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-pulse" />
        </div>
        <div className="text-center space-y-3 w-80">
          <h3 className="text-xl font-bold text-slate-800 thai-text">กำลังวิเคราะห์จัดหมวดหมู่อัจฉริยะ...</h3>
          <p className="text-slate-500 thai-text text-sm">{statusMsg}</p>
          <div className="w-full bg-slate-100 rounded-full h-2">
            <motion.div 
              className="h-2 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.4 }}
            />
          </div>
          <p className="text-xs text-slate-400">{progress}%</p>
        </div>
      </div>
    )
  }

  const pendingCount = suggestions.filter(s => s._status === 'pending').length
  const highConfidenceCount = suggestions.filter(s => s._confidence >= 0.8).length

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
            onClick={runCategorization}
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
            <TagsIcon className="w-8 h-8 text-indigo-600" />
            รีวิวการจัดหมวดหมู่ (Categorization Review)
          </h2>
          <p className="text-slate-500 font-medium thai-text">ตรวจสอบและยืนยันหมวดหมู่ที่ระบบแนะนำ{!isMock && ' — ข้อมูลจากระบบวิเคราะห์จริง'}</p>
        </div>
        <div className="flex gap-4">
          <div className="bg-indigo-50 px-6 py-3 rounded-2xl text-center">
            <div className="text-2xl font-black text-indigo-600">{suggestions.length}</div>
            <div className="text-xs font-black text-indigo-900 uppercase tracking-widest thai-text">ทั้งหมด</div>
          </div>
          <div className="bg-emerald-50 px-6 py-3 rounded-2xl text-center">
            <div className="text-2xl font-black text-emerald-600">{highConfidenceCount}</div>
            <div className="text-xs font-black text-emerald-900 uppercase tracking-widest thai-text">มั่นใจสูง</div>
          </div>
          <div className="bg-amber-50 px-6 py-3 rounded-2xl text-center">
            <div className="text-2xl font-black text-amber-600">{pendingCount}</div>
            <div className="text-xs font-black text-amber-900 uppercase tracking-widest thai-text">รอตรวจสอบ</div>
          </div>
        </div>
      </div>

      {/* Batch Actions */}
      <div className="flex items-center justify-between bg-slate-50 p-4 rounded-2xl border border-slate-200">
        <label className="flex items-center space-x-3 cursor-pointer">
          <input
            type="checkbox"
            checked={selectedIds.size === suggestions.length && suggestions.length > 0}
            onChange={toggleSelectAll}
            className="w-5 h-5 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
          />
          <span className="font-bold text-slate-700 thai-text">
            เลือกทั้งหมด ({selectedIds.size}/{suggestions.length})
          </span>
        </label>

        <AnimatePresence>
          {selectedIds.size > 0 && (
            <motion.div 
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              className="flex items-center gap-3"
            >
              <button
                onClick={() => handleBatchAction('approve')}
                className="px-6 py-2 bg-emerald-500 text-white rounded-xl hover:bg-emerald-600 font-bold transition-colors flex items-center gap-2 shadow-lg shadow-emerald-500/20"
              >
                <CheckIcon className="w-5 h-5" />
                <span className="thai-text">อนุมัติ ({selectedIds.size})</span>
              </button>
              
              <button
                onClick={() => handleBatchAction('reject')}
                className="px-6 py-2 bg-rose-500 text-white rounded-xl hover:bg-rose-600 font-bold transition-colors flex items-center gap-2 shadow-lg shadow-rose-500/20"
              >
                <XIcon className="w-5 h-5" />
                <span className="thai-text">ปฏิเสธ ({selectedIds.size})</span>
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* List */}
      <div className="space-y-4 max-h-[600px] overflow-y-auto pr-2 pb-4">
        {suggestions.map((suggestion, index) => (
          <div
            key={index}
            className={`bg-white border-2 rounded-2xl transition-all ${
              suggestion._status === 'approve' 
                ? 'border-emerald-200 bg-emerald-50/30' 
                : suggestion._status === 'reject'
                ? 'border-rose-200 bg-rose-50/30'
                : selectedIds.has(index) 
                  ? 'border-indigo-300 shadow-md ring-4 ring-indigo-500/10' 
                  : 'border-slate-100 hover:border-slate-300'
            }`}
          >
            <div className="p-5 flex items-center gap-6">
              <input
                type="checkbox"
                checked={selectedIds.has(index)}
                onChange={() => toggleSelection(index)}
                className="w-6 h-6 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500 cursor-pointer"
              />
              
              <div className="flex-1 min-w-0 grid grid-cols-12 gap-4 items-center">
                <div className="col-span-5">
                  <h3 className="text-lg font-bold text-slate-800 truncate thai-text">
                    {suggestion._cleaned_name}
                  </h3>
                  <p className="text-sm text-slate-500 truncate thai-text mt-1">{suggestion.description || 'ไม่มีรายละเอียด'}</p>
                </div>
                
                <div className="col-span-5">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-bold text-slate-400 thai-text">หมวดหมู่แนะนำ:</span>
                    <span className="font-bold text-indigo-700 bg-indigo-50 px-3 py-1 rounded-lg thai-text">
                      {suggestion._suggested_category}
                    </span>
                  </div>
                </div>

                <div className="col-span-2 text-right">
                  <span className={`px-3 py-1.5 text-sm font-black rounded-full inline-block min-w-[70px] text-center ${
                    suggestion._confidence >= 0.8 ? 'bg-emerald-100 text-emerald-700' :
                    suggestion._confidence >= 0.6 ? 'bg-amber-100 text-amber-700' :
                    'bg-rose-100 text-rose-700'
                  }`}>
                    {Math.round(suggestion._confidence * 100)}%
                  </span>
                </div>
              </div>

              {suggestion._status !== 'pending' && (
                <div className="pl-4 border-l-2 border-slate-100">
                  {suggestion._status === 'approve' ? (
                    <div className="flex items-center gap-1 text-emerald-600 font-bold text-sm bg-emerald-50 px-3 py-1.5 rounded-xl">
                      <CheckCircleIcon className="w-5 h-5" /> อนุมัติ
                    </div>
                  ) : (
                    <div className="flex items-center gap-1 text-rose-600 font-bold text-sm bg-rose-50 px-3 py-1.5 rounded-xl">
                      <XIcon className="w-5 h-5" /> ปฏิเสธ
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Footer Actions */}
      <div className="flex items-center justify-between pt-8 border-t border-slate-100">
        <button
          onClick={onBack}
          className="px-8 py-4 text-slate-500 hover:text-slate-900 font-bold text-sm transition-all thai-text"
        >
          ← ย้อนกลับ
        </button>

        <button
          onClick={handleComplete}
          className="px-10 py-4 bg-indigo-600 hover:bg-indigo-700 text-white rounded-[2rem] font-black text-sm tracking-wide shadow-xl shadow-indigo-600/30 transition-all active:scale-95 flex items-center gap-3 thai-text"
        >
          สรุปผลและนำเข้าคลังข้อมูล
          <ArrowRightIcon className="w-5 h-5" />
        </button>
      </div>
    </div>
  )
}
