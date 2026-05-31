'use client'

import { useState, useCallback, useMemo, useEffect } from 'react'
import { 
  CheckCircle, 
  XCircle, 
  AlertCircle, 
  RefreshCcwIcon,
  SparklesIcon,
  ScaleIcon,
  CheckIcon,
  DatabaseIcon,
  SlidersIcon,
  Tag,
  Zap
} from 'lucide-react'
import { motion } from 'framer-motion'
import { toast } from 'react-hot-toast'

interface WorkflowStep {
  id: number
  name: string
  description: string
  status: 'pending' | 'processing' | 'completed' | 'error'
  count?: number
}

interface InternalPair {
  id_a: string
  name_a: string
  sku_a: string
  category_a: string
  category_id_a: string
  id_b: string
  name_b: string
  sku_b: string
  category_b: string
  category_id_b: string
  similarity: number
}

export default function DeduplicationTab({ onComplete }: { onComplete?: () => void }) {
  // Common states
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [currentStep, setCurrentStep] = useState(1)
  const [fallbackWarning, setFallbackWarning] = useState<{ active: boolean, messages: string[] } | null>(null)

  // Internal Deduplication states
  const [internalPairs, setInternalPairs] = useState<InternalPair[]>([])
  const [totalScanned, setTotalScanned] = useState(0)
  const [scanThreshold, setScanThreshold] = useState(0.95)
  const [scanLimit, setScanLimit] = useState(100)
  const [internalCurrentIndex, setInternalCurrentIndex] = useState(0)
  const [resolvedCount, setResolvedCount] = useState({ merged: 0, kept: 0 })

  const workflowSteps = useMemo<WorkflowStep[]>(() => {
    return [
      { id: 1, name: 'System Scan', description: 'Scan Database Baseline', status: currentStep > 1 ? 'completed' : currentStep === 1 ? 'processing' : 'pending' },
      { id: 2, name: 'AI Analyze', description: 'Calculating Similarities', status: currentStep > 2 ? 'completed' : currentStep === 2 ? 'processing' : 'pending' },
      { id: 3, name: 'Database Audit', description: 'Audit & Resolve Pairs', status: currentStep > 3 ? 'completed' : currentStep === 3 ? 'processing' : 'pending' },
      { id: 4, name: 'Audit Summary', description: 'Results & Catalog Health', status: currentStep === 4 ? 'processing' : 'pending' }
    ]
  }, [currentStep])

  // --- INTERNAL DEDUPLICATION UTILITIES ---
  const scanInternalCatalog = useCallback(async () => {
    setLoading(true)
    setError(null)
    setCurrentStep(2)
    setInternalCurrentIndex(0)
    setResolvedCount({ merged: 0, kept: 0 })
    setFallbackWarning(null)
    
    try {
      const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'
      const response = await fetch(`${apiBase}/api/v1/match/scan-internal`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          threshold: scanThreshold,
          limit: scanLimit
        })
      })

      if (!response.ok) {
        const errText = await response.text()
        throw new Error(`Failed to scan: ${response.status} - ${errText}`)
      }

      const data = await response.json()
      setTotalScanned(data.total_scanned)
      setInternalPairs(data.results || [])
      
      if (data.fallback_active) {
        setFallbackWarning({ active: true, messages: data.system_warnings || [] })
        setCurrentStep(3)
        return
      }
      
      // If pairs found, go to audit step, else go straight to summary
      if (data.results && data.results.length > 0) {
        setCurrentStep(3)
        toast.success(`ตรวจพบสินค้าซ้ำแฝง ${data.pairs_found} คู่ ดึงข้อมูลเริ่มการตรวจสอบ!`)
      } else {
        setCurrentStep(4)
        toast.success('ยอดเยี่ยมมาก! ไม่พบสินค้าซ้ำเชิงเวกเตอร์ในระบบเดิมเลย!')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Internal scan failed')
      setCurrentStep(1)
      toast.error('เกิดข้อผิดพลาดในการเชื่อมต่อเซิร์ฟเวอร์สแกน')
    } finally {
      setLoading(false)
    }
  }, [scanThreshold, scanLimit])

  const triggerMLRetrain = useCallback(async () => {
    setLoading(true)
    try {
      const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'
      const response = await fetch(`${apiBase}/api/v1/learn/retrain`, {
        method: 'POST',
      })
      if (!response.ok) {
        const errJson = await response.json().catch(() => null)
        throw new Error(errJson?.detail || `Error: ${response.status}`)
      }
      toast.success('สอน AI สำเร็จ! โมเดลเรียนรู้การแยกแยะสินค้าซ้ำจากประวัติการรีวิวแล้ว')
    } catch (err) {
      toast.error('ไม่สามารถสอน AI ได้: ' + (err instanceof Error ? err.message : String(err)))
    } finally {
      setLoading(false)
    }
  }, [])

  const resolveInternalPair = useCallback(async (action: 'merge' | 'keep_both', keepSide?: 'a' | 'b') => {
    const pair = internalPairs[internalCurrentIndex]
    if (!pair) return

    setLoading(true)
    try {
      const keepId = action === 'merge' ? (keepSide === 'a' ? pair.id_a : pair.id_b) : undefined
      const response = await fetch('/api/similarity/resolve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action,
          id_a: pair.id_a,
          id_b: pair.id_b,
          keep_id: keepId,
          similarity: pair.similarity
        })
      })

      if (!response.ok) {
        const errJson = await response.json()
        throw new Error(errJson.error || 'Failed to resolve duplicate')
      }

      // Success
      toast.success(
        action === 'merge' 
          ? `ยุบรวมสำเร็จ! อนุมัติ: ${keepSide === 'a' ? pair.name_a : pair.name_b}` 
          : 'บันทึกเรียบร้อย: ยืนยันเป็นคนละสินค้ากัน'
      )

      // Update counters
      setResolvedCount(prev => ({
        merged: prev.merged + (action === 'merge' ? 1 : 0),
        kept: prev.kept + (action === 'keep_both' ? 1 : 0)
      }))

      // Move forward
      if (internalCurrentIndex < internalPairs.length - 1) {
        setInternalCurrentIndex(prev => prev + 1)
      } else {
        setCurrentStep(4) // Move to summary when complete
      }

    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'เกิดข้อผิดพลาดในการสะสาง')
    } finally {
      setLoading(false)
    }
  }, [internalPairs, internalCurrentIndex])

  // Keyboard shortcuts listener for Internal Catalog Review
  useEffect(() => {
    if (currentStep !== 3 || !internalPairs[internalCurrentIndex]) return

    const handleKeyDown = (e: KeyboardEvent) => {
      // Avoid triggering when loading or when user is typing in inputs or textareas
      if (loading) return
      
      if (document.activeElement?.tagName === 'INPUT' || document.activeElement?.tagName === 'TEXTAREA') {
        return
      }

      const key = e.key.toLowerCase()
      const code = e.code

      if (code === 'KeyA' || code === 'ArrowLeft' || key === 'a' || key === 'ฟ' || key === 'ฤ') {
        e.preventDefault()
        resolveInternalPair('merge', 'a')
      } else if (code === 'KeyD' || code === 'ArrowRight' || key === 'd' || key === 'ก' || key === 'ฏ') {
        e.preventDefault()
        resolveInternalPair('merge', 'b')
      } else if (code === 'KeyS' || code === 'ArrowDown' || key === 's' || key === 'ห' || key === 'ฆ') {
        e.preventDefault()
        resolveInternalPair('keep_both')
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [currentStep, internalPairs, internalCurrentIndex, loading, resolveInternalPair])

  return (
    <div className="h-full bg-gray-50 font-sans flex flex-col overflow-hidden">
      {/* Top Title Bar */}
      <div className="bg-white border-b border-slate-200 px-8 py-4 z-20 relative shadow-sm flex items-center justify-between">
        <div className="flex items-center gap-3">
          <DatabaseIcon className="w-5 h-5 text-indigo-600" />
          <span className="text-sm font-black text-slate-800 uppercase tracking-wider thai-text">
            ระบบตรวจสอบและชำระคุณภาพคลังสินค้า (Catalog Health Audit)
          </span>
        </div>
        <div className="flex gap-4 text-xs font-black uppercase text-slate-400">
           <span>คลังสินค้าหลัก: <strong className="text-slate-700">3,103 approved</strong></span>
        </div>
      </div>

      <div className="flex-1 overflow-x-hidden overflow-y-auto p-8 relative">
        {/* Decorative Background Elements */}
        <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-indigo-50/40 rounded-full blur-[120px] -mr-64 -mt-64 pointer-events-none" />
        <div className="absolute bottom-10 left-10 w-[400px] h-[400px] bg-emerald-50/40 rounded-full blur-[120px] pointer-events-none" />

        <div className="max-w-7xl mx-auto relative z-10">
          {/* Header */}
          <div className="mb-12">
            <h1 className="text-3xl font-black text-slate-900 tracking-tight uppercase thai-text flex items-center gap-4">
              Catalog Audit & Resolution
            </h1>
            <p className="text-slate-500 mt-2 font-medium thai-text leading-relaxed max-w-3xl">
              เครื่องมือตรวจสอบคุณภาพคลังข้อมูลภายใน สแกนหาและสมานสินค้าซ้ำแฝง 3,103 รายการ เพื่อสร้างคลังความรู้สอนระบบที่บริสุทธิ์ 100%
            </p>
          </div>

          {/* Workflow Stepper */}
          <div className="premium-card p-10 mb-12 bg-white/40 border-white/80 overflow-hidden">
             <div className="relative flex items-center justify-between">
                <div className="absolute top-6 left-0 right-0 h-1 bg-slate-100 rounded-full overflow-hidden">
                   <motion.div 
                      initial={{ width: 0 }}
                      animate={{ width: `${((currentStep - 1) / (workflowSteps.length - 1)) * 100}%` }}
                      className="h-full bg-indigo-500 shadow-[0_0_12px_rgba(99,102,241,0.5)]"
                   />
                </div>
                {workflowSteps.map((step) => (
                   <div key={step.id} className="relative z-10 flex flex-col items-center group">
                      <div className={`w-14 h-14 rounded-2xl flex items-center justify-center border-4 transition-all duration-500 ${
                         currentStep > step.id ? 'bg-emerald-500 border-white text-white shadow-lg shadow-emerald-200 rotate-[360deg]' :
                         currentStep === step.id ? 'bg-indigo-600 border-white text-white shadow-xl shadow-indigo-200 scale-110' :
                         'bg-white border-slate-50 text-slate-300'
                      }`}>
                         {currentStep > step.id ? <CheckCircle className="w-6 h-6" /> : <span className="text-lg font-black">{step.id}</span>}
                      </div>
                      <div className="mt-4 text-center">
                         <h4 className={`text-xs font-black uppercase tracking-[0.15em] mb-1 transition-colors ${currentStep >= step.id ? 'text-slate-900' : 'text-slate-300'}`}>
                            {step.name}
                         </h4>
                         <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest opacity-0 group-hover:opacity-100 transition-opacity">
                            {step.description}
                         </p>
                      </div>
                   </div>
                ))}
             </div>
          </div>

          {error && (
            <div className="premium-card p-6 border-rose-100 bg-rose-50/50 mb-12 flex items-center gap-4 text-rose-600 thai-text">
               <AlertCircle className="w-6 h-6 flex-shrink-0" />
               <div>
                  <p className="font-black text-sm uppercase">พบข้อผิดพลาดในระบบ</p>
                  <p className="text-xs font-medium mt-1">{error}</p>
               </div>
            </div>
          )}

          {/* Main Action Zone */}
          <div className="mb-16">
            {currentStep === 1 && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                 {/* Left Settings Panel */}
                 <div className="premium-card p-8 bg-white/60 border-white h-fit">
                    <div className="flex items-center gap-3 mb-6 pb-4 border-b border-slate-100">
                       <SlidersIcon className="w-5 h-5 text-indigo-600" />
                       <h3 className="text-base font-black text-slate-900 uppercase tracking-wider">เกณฑ์ความคล่องตัว</h3>
                    </div>

                    <div className="space-y-8">
                       {/* Threshold slider */}
                       <div className="space-y-2">
                          <div className="flex justify-between items-baseline">
                             <label className="text-[10px] font-black text-slate-400 uppercase tracking-wider">Similarity Threshold</label>
                             <span className="text-sm font-black text-indigo-600">{(scanThreshold * 100).toFixed(0)}%</span>
                          </div>
                          <input 
                             type="range" 
                             min="0.80" 
                             max="1.00" 
                             step="0.01" 
                             value={scanThreshold} 
                             onChange={(e) => setScanThreshold(parseFloat(e.target.value))}
                             className="w-full h-1 bg-slate-100 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                          />
                          <p className="text-[10px] text-slate-400 font-bold leading-normal thai-text">
                             ค่าความเหมือนของเวกเตอร์ที่ยอมให้จับเป็นคู่ซ้ำ (แนะนำ 95% ขึ้นไป)
                          </p>
                       </div>

                       {/* Limit Input */}
                       <div className="space-y-2">
                          <div className="flex justify-between items-baseline">
                             <label className="text-[10px] font-black text-slate-400 uppercase tracking-wider">Audit Limit</label>
                             <span className="text-sm font-black text-indigo-600">{scanLimit} คู่</span>
                          </div>
                          <input 
                             type="range" 
                             min="10" 
                             max="500" 
                             step="10" 
                             value={scanLimit} 
                             onChange={(e) => setScanLimit(parseInt(e.target.value))}
                             className="w-full h-1 bg-slate-100 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                          />
                          <p className="text-[10px] text-slate-400 font-bold leading-normal thai-text">
                             จำนวนคู่สูงสุดที่จะนำขึ้นมาตรวจสอบในรอบการรันนี้
                          </p>
                       </div>
                    </div>
                 </div>

                 {/* Right Scan Action Panel */}
                 <div className="lg:col-span-2 premium-card p-12 bg-white/40 border-white text-center flex flex-col items-center justify-center">
                    <div className="w-20 h-20 rounded-3xl bg-indigo-600 text-white flex items-center justify-center mb-8 shadow-2xl shadow-indigo-100">
                       <ScaleIcon className="w-10 h-10 animate-pulse" />
                    </div>
                    <h3 className="text-2xl font-black text-slate-900 mb-3 uppercase tracking-tight thai-text">เริ่มสแกนตรวจสอบความซ้ำซ้อนภายในคลังสินค้า</h3>
                    <p className="text-xs font-black text-slate-400 uppercase tracking-widest mb-10 leading-relaxed max-w-md">
                       ระบบจะคำนวณเวกเตอร์ความคล้ายคลึงเชิงความหมายแบบคู่ขนานบน NumPy กับสินค้า 3,103 รายการทั้งหมดในชั่วพริบตา
                    </p>

                    <button onClick={scanInternalCatalog} className="btn-premium w-full max-w-sm px-16 py-6 text-lg group h-auto mb-4">
                       <Zap className="w-5 h-5 mr-3 group-hover:animate-bounce" />
                       <span className="thai-text uppercase tracking-widest font-black">เริ่มการสแกนหาของซ้ำเชิงลึก</span>
                    </button>
                    
                    <button onClick={triggerMLRetrain} className="btn-premium-secondary w-full max-w-sm px-10 py-4 text-sm group h-auto bg-emerald-50 text-emerald-600 border-emerald-200 hover:bg-emerald-100 hover:border-emerald-300">
                       <SparklesIcon className="w-4 h-4 mr-2 group-hover:rotate-12 transition-transform" />
                       <span className="thai-text uppercase tracking-widest font-black">เริ่มสอน AI จากประวัติ (Retrain Model)</span>
                    </button>
                 </div>
              </div>
            )}

            {currentStep === 2 && (
              <div className="premium-card p-24 text-center flex flex-col items-center justify-center bg-white/40 border-white">
                <div className="w-20 h-20 relative flex items-center justify-center mb-8">
                  <div className="absolute inset-0 border-4 border-indigo-100 rounded-full" />
                  <div className="absolute inset-0 border-4 border-indigo-600 rounded-full border-t-transparent animate-spin" />
                </div>
                <h3 className="text-2xl font-black text-slate-900 thai-text uppercase tracking-tight">กำลังตรวจสอบสินค้าซ้ำ 3,103 รายการภายในฐานข้อมูล...</h3>
                <p className="text-xs font-bold text-slate-400 mt-2 uppercase tracking-widest">Scanning and computing pairwise similarities on GPU/CPU cache...</p>
              </div>
            )}

            {currentStep === 3 && fallbackWarning?.active && (
              <div className="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-sm flex items-center justify-center p-4">
                 <div className="bg-white rounded-3xl p-8 max-w-lg w-full shadow-2xl border border-amber-200 relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-32 h-32 bg-amber-500/10 rounded-full blur-2xl -mr-16 -mt-16" />
                    
                    <div className="w-16 h-16 bg-amber-100 rounded-2xl flex items-center justify-center mb-6 border border-amber-200">
                       <AlertCircle className="w-8 h-8 text-amber-600" />
                    </div>
                    
                    <h3 className="text-2xl font-black text-slate-900 mb-2 thai-text uppercase tracking-tight">ระบบ AI ไม่พร้อมใช้งาน</h3>
                    <p className="text-slate-600 thai-text text-sm mb-4 leading-relaxed font-medium">
                       ระบบพบปัญหาในการเชื่อมต่อกับ Machine Learning Model หลัก (Batch Inference) หรือโมเดลยังไม่ได้รับการเทรน
                    </p>
                    
                    <div className="bg-amber-50 rounded-xl p-4 mb-8">
                       <p className="text-xs font-black text-amber-800 uppercase tracking-widest mb-2">ข้อความจากระบบ (System Warning):</p>
                       <ul className="list-disc list-inside text-sm text-amber-700 thai-text">
                          {fallbackWarning.messages.map((msg, i) => (
                             <li key={i}>{msg}</li>
                          ))}
                       </ul>
                    </div>
                    
                    <p className="text-sm font-bold text-slate-700 thai-text mb-8 text-center bg-slate-50 p-4 rounded-xl border border-slate-100">
                       ระบบเปลี่ยนไปใช้ <strong>ระบบสำรอง (Cosine Similarity พื้นฐาน)</strong> อัตโนมัติ ผลลัพธ์อาจมีความแม่นยำลดลงเล็กน้อย คุณต้องการดำเนินการตรวจสอบต่อไปหรือไม่?
                    </p>
                    
                    <div className="flex gap-4">
                       <button 
                          onClick={() => { setFallbackWarning(null); setCurrentStep(1); setInternalPairs([]); }}
                          className="flex-1 px-6 py-4 rounded-2xl font-black text-slate-700 bg-white border-2 border-slate-200 hover:bg-slate-50 hover:border-slate-300 transition-colors thai-text uppercase text-sm"
                       >
                          ยกเลิก (รอ AI พร้อม)
                       </button>
                       <button 
                          onClick={() => setFallbackWarning({ ...fallbackWarning, active: false })}
                          className="flex-1 px-6 py-4 rounded-2xl font-black text-amber-900 bg-amber-400 hover:bg-amber-500 transition-colors thai-text uppercase shadow-lg shadow-amber-500/20 text-sm"
                       >
                          ยอมรับและทำต่อ
                       </button>
                    </div>
                 </div>
              </div>
            )}

            {currentStep === 3 && !fallbackWarning?.active && internalPairs[internalCurrentIndex] && (
              <div className="space-y-12 animate-in fade-in slide-in-from-bottom-12 duration-700">
                <div className="flex items-center justify-between">
                   <div>
                      <h3 className="text-2xl font-black text-slate-900 tracking-tight uppercase">Audit Database Duplicates</h3>
                      <p className="text-indigo-500 text-xs font-black uppercase tracking-[0.2em] mt-1">คลังข้อมูลสินค้าซ้ำในฐานข้อมูลหลัก</p>
                   </div>
                   <div className="px-6 py-2 bg-indigo-500 rounded-2xl text-white font-black text-sm shadow-lg shadow-indigo-100 tracking-tighter italic">
                      {internalCurrentIndex + 1} / {internalPairs.length} คู่
                   </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 relative">
                   <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-20 hidden lg:flex w-20 h-20 rounded-full bg-white shadow-2xl border border-slate-50 items-center justify-center">
                      <ScaleIcon className="w-8 h-8 text-indigo-500 animate-pulse" />
                   </div>

                   {[
                     { 
                       type: 'สินค้าฝั่ง A (ซ้าย)', 
                       name_th: internalPairs[internalCurrentIndex].name_a, 
                       sku: internalPairs[internalCurrentIndex].sku_a,
                       category: internalPairs[internalCurrentIndex].category_a,
                       color: 'indigo', 
                       side: 'a' as const
                     },
                     { 
                       type: 'สินค้าฝั่ง B (ขวา)', 
                       name_th: internalPairs[internalCurrentIndex].name_b, 
                       sku: internalPairs[internalCurrentIndex].sku_b,
                       category: internalPairs[internalCurrentIndex].category_b,
                       color: 'emerald', 
                       side: 'b' as const
                     }
                   ].map((item, i) => (
                     <div key={i} className={`premium-card p-0 overflow-hidden bg-white/80 border-slate-100 hover:border-indigo-50 hover:shadow-xl transition-all duration-300`}>
                        <div className={`px-6 py-3 bg-${i === 0 ? 'indigo-600' : 'emerald-600'} text-white flex justify-between items-center`}>
                           <span className="text-xs font-black uppercase tracking-[0.2em]">{item.type}</span>
                           <span className="text-xs font-black bg-white/20 px-3 py-1 rounded-lg">SKU: {item.sku}</span>
                        </div>
                        <div className="p-10 flex flex-col justify-between h-[220px]">
                           <p className="text-xl font-black text-slate-800 thai-text leading-relaxed tracking-tight">
                              {item.name_th}
                           </p>
                           <div className="pt-4 border-t border-slate-100 flex items-center justify-between">
                              <span className="flex items-center gap-1.5 text-xs font-black text-slate-400 uppercase tracking-widest">
                                 <Tag className="w-3.5 h-3.5 text-slate-400" />
                                 {item.category}
                              </span>
                           </div>
                        </div>
                     </div>
                   ))}
                </div>

                {/* Symmetric Control Center below the cards */}
                <div className="bg-slate-950 p-10 rounded-[48px] text-center shadow-2xl relative overflow-hidden">
                   <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-500/10 rounded-full blur-[80px] -mr-32 -mt-32" />
                   
                   <div className="inline-flex gap-2 items-center bg-white/5 border border-white/10 px-6 py-2 rounded-full mb-8">
                      <span className="text-[10px] font-black text-indigo-400 uppercase tracking-widest">Semantic Match:</span>
                      <span className="text-sm font-black text-white italic tracking-widest">{(internalPairs[internalCurrentIndex].similarity * 100).toFixed(1)}% Match</span>
                   </div>

                   <h4 className="text-lg font-black text-white thai-text mb-2 uppercase tracking-tight">การตัดสินใจสะสางข้อมูลซ้ำซ้อน</h4>
                   <p className="text-[10px] text-slate-400 font-bold uppercase tracking-widest mb-8">
                     สามารถใช้ปุ่มลัดคีย์บอร์ด [A] [S] [D] หรือปุ่มลูกศร [←] [↓] [→] เพื่อทำงานได้รวดเร็วยิ่งขึ้น
                   </p>

                   <div className="flex flex-col md:flex-row justify-center items-center gap-6">
                      {/* Merge A (Keep Left) */}
                      <button 
                         onClick={() => resolveInternalPair('merge', 'a')}
                         className="w-full md:w-auto bg-indigo-600 hover:bg-indigo-700 text-white shadow-lg shadow-indigo-500/10 px-8 py-4 rounded-[20px] font-black text-xs transition-all hover:scale-105 flex items-center justify-center gap-3"
                      >
                         <div className="flex items-center gap-1 bg-white/15 px-2 py-0.5 rounded text-[9px] font-mono border border-white/20 select-none">
                            <span>A</span>
                            <span>/</span>
                            <span>←</span>
                         </div>
                         <span className="thai-text uppercase font-black tracking-wide">👈 ยุบรวม: เก็บฝั่งซ้าย (A)</span>
                      </button>

                      {/* Keep Both / Different */}
                      <button 
                         onClick={() => resolveInternalPair('keep_both')}
                         className="w-full md:w-auto bg-slate-800 hover:bg-slate-700 text-white border border-slate-700 px-8 py-4 rounded-[20px] font-black text-xs transition-all hover:scale-105 flex items-center justify-center gap-3"
                      >
                         <div className="flex items-center gap-1 bg-white/15 px-2 py-0.5 rounded text-[9px] font-mono border border-white/20 select-none">
                            <span>S</span>
                            <span>/</span>
                            <span>↓</span>
                         </div>
                         <span className="thai-text uppercase font-black tracking-wide">🛡️ เป็นคนละสินค้า (Keep Both)</span>
                      </button>

                      {/* Merge B (Keep Right) */}
                      <button 
                         onClick={() => resolveInternalPair('merge', 'b')}
                         className="w-full md:w-auto bg-emerald-600 hover:bg-emerald-700 text-white shadow-lg shadow-emerald-500/10 px-8 py-4 rounded-[20px] font-black text-xs transition-all hover:scale-105 flex items-center justify-center gap-3"
                      >
                         <div className="flex items-center gap-1 bg-white/15 px-2 py-0.5 rounded text-[9px] font-mono border border-white/20 select-none">
                            <span>D</span>
                            <span>/</span>
                            <span>→</span>
                         </div>
                         <span className="thai-text uppercase font-black tracking-wide">👉 ยุบรวม: เก็บฝั่งขวา (B)</span>
                      </button>
                   </div>
                </div>
              </div>
            )}

            {currentStep === 4 && (
              <div className="premium-card p-16 bg-white/40 border-white text-center flex flex-col items-center justify-center animate-in fade-in zoom-in duration-500">
                 <div className="w-20 h-20 rounded-full bg-emerald-100 border-4 border-emerald-500 flex items-center justify-center mb-8 shadow-xl shadow-emerald-100">
                    <CheckIcon className="w-10 h-10 text-emerald-600" />
                 </div>

                 <h3 className="text-3xl font-black text-slate-900 mb-2 uppercase tracking-tight thai-text">เสร็จสิ้นภารกิจตรวจสอบคุณภาพ!</h3>
                 <p className="text-xs font-black text-slate-400 uppercase tracking-widest mb-10 leading-relaxed max-w-md">
                    คุณกานสะสางความซ้ำซ้อนของข้อมูลคลังสอนระบบ baseline ปัจจุบันเรียบร้อยแล้ว
                 </p>

                 <div className="grid grid-cols-2 gap-8 w-full max-w-md mb-12 p-6 bg-white rounded-3xl border border-slate-100 shadow-sm">
                    <div className="text-center">
                       <p className="text-sm font-black text-indigo-600">{resolvedCount.merged} รายการ</p>
                       <p className="text-[10px] font-black text-slate-400 uppercase tracking-wider mt-1">สินค้าที่กดยุบรวม (Merged)</p>
                    </div>
                    <div className="text-center">
                       <p className="text-sm font-black text-emerald-600">{resolvedCount.kept} คู่</p>
                       <p className="text-[10px] font-black text-slate-400 uppercase tracking-wider mt-1">คู่อลุ่มอล่วย (Kept Both)</p>
                    </div>
                 </div>

                 <button 
                    onClick={() => { setCurrentStep(1); setInternalPairs([]); }}
                    className="btn-premium px-16 py-6 group h-auto"
                 >
                    <RefreshCcwIcon className="w-5 h-5 mr-3 group-hover:rotate-180 transition-transform duration-700" />
                    <span className="thai-text uppercase tracking-widest font-black">สแกนตรวจสอบใหม่อีกครั้ง</span>
                 </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
