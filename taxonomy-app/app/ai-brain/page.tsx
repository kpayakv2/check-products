'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  BrainCircuitIcon, 
  TargetIcon, 
  DatabaseIcon, 
  ActivityIcon,
  SparklesIcon,
  RefreshCwIcon,
  AlertTriangleIcon,
  CheckCircle2Icon,
  HistoryIcon,
  TrendingUpIcon,
  TrendingDownIcon,
  MinusIcon,
  FilterIcon,
  SearchIcon,
  CpuIcon,
  ShieldCheckIcon,
  ArrowRightIcon,
  FingerprintIcon,
  HashIcon,
  TextIcon,
  CheckIcon,
  XIcon
} from 'lucide-react'
import { toast } from 'react-hot-toast'
import Sidebar from '@/components/Layout/Sidebar'
import Header from '@/components/Layout/Header'

interface FeatureImportance {
  feature: string
  importance: number
}

interface MLStatus {
  is_trained: boolean
  message?: string
  accuracy?: number
  total_samples?: number
  average_confidence?: number
  feature_importance?: FeatureImportance[]
}

interface TrainingSession {
  id: string
  training_date: string
  model_type: string
  total_samples: number
  train_accuracy: number
  test_accuracy: number
  cv_mean_accuracy: number
  feature_importance: FeatureImportance[]
  classes: string[]
}

interface TrainingHistory {
  status: string
  history: TrainingSession[]
  total: number
  source?: string
}

const formatFeatureName = (feature: string) => {
  const map: Record<string, { label: string, icon: any, desc: string }> = {
    'num_similarity': { label: 'เปรียบเทียบตัวเลข (Number Match)', icon: HashIcon, desc: 'ตรวจสอบรหัสรุ่น หรือขนาดบรรจุภัณฑ์ที่แฝงในชื่อ' },
    'brand_similarity': { label: 'เปรียบเทียบแบรนด์ (Brand Match)', icon: ShieldCheckIcon, desc: 'ตรวจสอบชื่อยี่ห้อที่ซ่อนอยู่' },
    'len_diff': { label: 'ความยาวอักษร (Length Diff)', icon: TextIcon, desc: 'ความต่างของความยาวชื่อสินค้า' },
    'similarity_score': { label: 'ความหมายเชิงลึก (Vector Similarity)', icon: FingerprintIcon, desc: 'ความคล้ายคลึงเวกเตอร์ 384-มิติ' },
  }
  return map[feature] || { label: feature, icon: TargetIcon, desc: 'ฟีเจอร์ตัวแปรอื่นๆ' }
}

export default function AIBrainDashboard() {
  const [status, setStatus] = useState<MLStatus | null>(null)
  const [history, setHistory] = useState<TrainingHistory | null>(null)
  const [loading, setLoading] = useState(true)
  const [retraining, setRetraining] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  // Animation states for pipeline
  const [activeStep, setActiveStep] = useState(0)

  useEffect(() => {
    const timer = setInterval(() => {
      setActiveStep((prev) => (prev + 1) % 4)
    }, 2500)
    return () => clearInterval(timer)
  }, [])

  const fetchStatus = async () => {
    try {
      setLoading(true)
      const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'
      const [statusRes, historyRes] = await Promise.all([
        fetch(`${apiBase}/api/v1/learn/status`),
        fetch(`${apiBase}/api/v1/learn/history?limit=10`)
      ])
      if (!statusRes.ok) throw new Error('Failed to fetch ML status')
      const data = await statusRes.json()
      setStatus(data)
      if (historyRes.ok) {
        const hData = await historyRes.json()
        setHistory(hData)
      }
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchStatus()
  }, [])

  const handleRetrain = async () => {
    setRetraining(true)
    try {
      const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'
      const response = await fetch(`${apiBase}/api/v1/learn/retrain`, {
        method: 'POST',
      })
      if (!response.ok) {
        const errJson = await response.json().catch(() => null)
        throw new Error(errJson?.detail || `Error: ${response.status}`)
      }
      toast.success('สอน AI สำเร็จ! โมเดลอัปเดตความรู้เรียบร้อยแล้ว')
      await fetchStatus()
    } catch (err) {
      toast.error('ไม่สามารถสอน AI ได้: ' + (err instanceof Error ? err.message : String(err)))
    } finally {
      setRetraining(false)
    }
  }

  if (loading && !status) {
    return (
      <div className="flex h-screen bg-slate-950">
        <Sidebar />
        <div className="flex-1 flex flex-col overflow-hidden">
          <Header />
          <div className="flex-1 flex items-center justify-center">
            <div className="flex flex-col items-center space-y-6">
              <div className="relative">
                <div className="absolute inset-0 bg-indigo-500 rounded-full blur-xl opacity-20 animate-pulse"></div>
                <BrainCircuitIcon className="h-20 w-20 text-indigo-400 relative z-10 animate-bounce" />
              </div>
              <p className="thai-text text-xl text-indigo-200 font-medium tracking-wide">กำลังเชื่อมต่อสมองกล AI...</p>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-screen bg-slate-50 dark:bg-slate-900 transition-colors duration-300">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-y-auto p-4 sm:p-6 lg:p-8">
          <div className="space-y-8 max-w-7xl mx-auto">
            
            {/* Header Premium Section */}
            <motion.div 
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              className="relative overflow-hidden rounded-[2.5rem] bg-slate-900 shadow-2xl border border-slate-800"
            >
              {/* Background Glow */}
              <div className="absolute top-0 -left-1/4 w-1/2 h-full bg-indigo-500/20 blur-[100px] pointer-events-none rounded-full transform -rotate-12"></div>
              <div className="absolute bottom-0 -right-1/4 w-1/2 h-full bg-purple-500/20 blur-[100px] pointer-events-none rounded-full transform rotate-12"></div>
              
              <div className="relative p-10 md:p-14 flex flex-col md:flex-row items-center justify-between gap-8 z-10">
                <div className="flex items-center gap-8">
                  <div className="relative group">
                    <div className="absolute inset-0 bg-indigo-500 rounded-3xl blur-xl opacity-50 group-hover:opacity-75 transition-opacity duration-500"></div>
                    <div className="bg-slate-900/80 p-5 rounded-3xl backdrop-blur-xl border border-slate-700/50 relative">
                      <BrainCircuitIcon className="h-14 w-14 text-indigo-300" />
                    </div>
                  </div>
                  <div>
                    <h1 className="text-4xl md:text-5xl font-black text-white thai-text tracking-tight mb-3">สมองกล AI <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-purple-400">อัจฉริยะ</span></h1>
                    <p className="text-slate-400 thai-text text-lg md:text-xl font-medium max-w-xl">
                      ศูนย์ควบคุม Secondary Smart Filter (ML Feedback) สำหรับคัดกรองคู่สินค้าหลอกทิ้งอย่างแม่นยำ
                    </p>
                  </div>
                </div>
                
                {status?.is_trained && (
                  <div className="hidden md:flex gap-4">
                    <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-4 text-center min-w-[120px]">
                      <div className="text-slate-400 text-sm thai-text mb-1">ความแม่นยำ</div>
                      <div className="text-3xl font-black text-emerald-400">{status.accuracy}%</div>
                    </div>
                    <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-4 text-center min-w-[120px]">
                      <div className="text-slate-400 text-sm thai-text mb-1">คู่ที่เรียนรู้</div>
                      <div className="text-3xl font-black text-blue-400">{status.total_samples}</div>
                    </div>
                  </div>
                )}
              </div>
            </motion.div>

            {error && (
              <div className="bg-red-500/10 border border-red-500/30 p-5 rounded-2xl flex items-start backdrop-blur-sm">
                <AlertTriangleIcon className="w-6 h-6 text-red-400 mr-4 flex-shrink-0 mt-0.5" />
                <p className="text-red-400 thai-text text-lg">{error}</p>
              </div>
            )}

            {status?.is_trained === false ? (
              <motion.div 
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="bg-slate-800/50 border border-slate-700/50 rounded-[2.5rem] p-16 text-center backdrop-blur-xl"
              >
                <div className="relative inline-block mb-8">
                  <div className="absolute inset-0 bg-slate-600 rounded-full blur-2xl opacity-20"></div>
                  <BrainCircuitIcon className="w-32 h-32 text-slate-600 relative z-10 mx-auto" />
                </div>
                <h2 className="text-3xl font-bold text-slate-200 thai-text mb-4">สมองกลยังไม่ถูกปลุก</h2>
                <p className="text-slate-400 thai-text text-xl mb-10 max-w-2xl mx-auto leading-relaxed">
                  ระบบนี้ทำงานเป็นตัวกรองชั้นที่สอง (Secondary Filter) แต่ปัจจุบันยังไม่มีข้อมูลการสอน (Human Feedback) มากพอ โปรดเริ่มใช้งานสแกนสินค้าเพื่อเพิ่มแนวข้อสอบ
                </p>
                <button 
                  onClick={handleRetrain}
                  disabled={retraining}
                  className="btn-premium bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-400 hover:to-purple-500 text-white px-12 py-5 rounded-2xl text-xl shadow-[0_0_40px_rgba(99,102,241,0.4)] transition-all duration-300 hover:scale-105"
                >
                  {retraining ? (
                    <span className="flex items-center"><RefreshCwIcon className="w-6 h-6 mr-3 animate-spin" /> กำลังประมวลผล...</span>
                  ) : (
                    <span className="flex items-center"><SparklesIcon className="w-6 h-6 mr-3" /> ลองปลุกสมองกลทันที</span>
                  )}
                </button>
              </motion.div>
            ) : (
              <div className="grid grid-cols-1 xl:grid-cols-12 gap-8">
                
                {/* Left Column (Pipeline & Controls) */}
                <div className="xl:col-span-8 space-y-8">
                  
                  {/* Smart Filter Pipeline Animation */}
                  <motion.div 
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="bg-white dark:bg-slate-800 rounded-[2rem] p-8 shadow-xl border border-slate-200 dark:border-slate-700 relative overflow-hidden"
                  >
                    <div className="flex items-center gap-4 mb-10">
                      <div className="bg-indigo-100 dark:bg-indigo-900/50 p-3 rounded-2xl">
                        <FilterIcon className="w-7 h-7 text-indigo-600 dark:text-indigo-400" />
                      </div>
                      <div>
                        <h2 className="text-2xl font-bold text-slate-800 dark:text-white thai-text">กระบวนการกรองอัจฉริยะ (Smart Filter Pipeline)</h2>
                        <p className="text-slate-500 dark:text-slate-400 thai-text mt-1">วิธีที่ AI แยกแยะ "คู่สินค้าหลอก" ออกจากคู่สินค้าจริง</p>
                      </div>
                    </div>

                    <div className="relative px-4 py-8">
                      {/* Connection Line */}
                      <div className="absolute top-1/2 left-8 right-8 h-1 bg-slate-100 dark:bg-slate-700 -translate-y-1/2 rounded-full z-0 overflow-hidden">
                        <motion.div 
                          className="h-full bg-gradient-to-r from-transparent via-indigo-500 to-transparent w-1/3"
                          animate={{ x: ['-100%', '300%'] }}
                          transition={{ repeat: Infinity, duration: 2.5, ease: "linear" }}
                        />
                      </div>

                      <div className="grid grid-cols-3 gap-6 relative z-10">
                        {/* Step 1 */}
                        <div className={`flex flex-col items-center text-center transition-all duration-500 ${activeStep === 0 || activeStep === 3 ? 'scale-110 opacity-100' : 'scale-95 opacity-50'}`}>
                          <div className={`w-20 h-20 rounded-2xl flex items-center justify-center mb-4 shadow-lg transition-colors duration-500
                            ${activeStep === 0 || activeStep === 3 ? 'bg-blue-500 text-white' : 'bg-white dark:bg-slate-700 text-slate-400'}`}>
                            <SearchIcon className="w-10 h-10" />
                          </div>
                          <h3 className="font-bold text-slate-800 dark:text-white thai-text text-lg">ด่านแรก</h3>
                          <p className="text-sm text-slate-500 dark:text-slate-400 thai-text mt-2">ดึงคู่ที่คล้ายกันด้วย Vector Search</p>
                        </div>
                        
                        {/* Step 2 */}
                        <div className={`flex flex-col items-center text-center transition-all duration-500 ${activeStep === 1 || activeStep === 3 ? 'scale-110 opacity-100' : 'scale-95 opacity-50'}`}>
                          <div className={`w-20 h-20 rounded-2xl flex items-center justify-center mb-4 shadow-lg transition-colors duration-500
                            ${activeStep === 1 || activeStep === 3 ? 'bg-indigo-500 text-white' : 'bg-white dark:bg-slate-700 text-slate-400'}`}>
                            <CpuIcon className="w-10 h-10" />
                          </div>
                          <h3 className="font-bold text-slate-800 dark:text-white thai-text text-lg">ดึงฟีเจอร์</h3>
                          <p className="text-sm text-slate-500 dark:text-slate-400 thai-text mt-2">เทียบแบรนด์ ตัวเลข ความยาว</p>
                        </div>

                        {/* Step 3 */}
                        <div className={`flex flex-col items-center text-center transition-all duration-500 ${activeStep === 2 || activeStep === 3 ? 'scale-110 opacity-100' : 'scale-95 opacity-50'}`}>
                          <div className={`w-20 h-20 rounded-2xl flex items-center justify-center mb-4 shadow-lg transition-colors duration-500
                            ${activeStep === 2 || activeStep === 3 ? 'bg-emerald-500 text-white' : 'bg-white dark:bg-slate-700 text-slate-400'}`}>
                            <ShieldCheckIcon className="w-10 h-10" />
                          </div>
                          <h3 className="font-bold text-slate-800 dark:text-white thai-text text-lg">คัดกรอง</h3>
                          <p className="text-sm text-slate-500 dark:text-slate-400 thai-text mt-2">หากมั่นใจว่าคนละตัว ปัดตกทันที!</p>
                        </div>
                      </div>
                    </div>
                  </motion.div>

                  {/* Feature Importance List */}
                  <motion.div 
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                    className="bg-white dark:bg-slate-800 rounded-[2rem] p-8 shadow-xl border border-slate-200 dark:border-slate-700"
                  >
                    <div className="flex items-center mb-8">
                      <div className="bg-purple-100 dark:bg-purple-900/50 p-3 rounded-2xl mr-4">
                        <SparklesIcon className="w-7 h-7 text-purple-600 dark:text-purple-400" />
                      </div>
                      <div>
                        <h2 className="text-2xl font-bold text-slate-800 dark:text-white thai-text">ฟีเจอร์ที่ใช้ตัดสินใจ (Feature Importance)</h2>
                        <p className="text-slate-500 dark:text-slate-400 thai-text">ปัจจัยที่โมเดลให้ความสำคัญมากที่สุดในการแยกคู่สินค้า</p>
                      </div>
                    </div>
                    
                    <div className="space-y-5">
                      {status?.feature_importance?.slice(0, 5).map((feat, index) => {
                        const featureData = formatFeatureName(feat.feature)
                        const Icon = featureData.icon
                        return (
                          <div key={index} className="group relative bg-slate-50 dark:bg-slate-900/50 p-4 rounded-2xl border border-slate-100 dark:border-slate-800 hover:border-indigo-300 dark:hover:border-indigo-700 transition-colors">
                            <div className="flex items-center justify-between mb-3">
                              <div className="flex items-center gap-3">
                                <div className="bg-white dark:bg-slate-800 p-2 rounded-xl shadow-sm">
                                  <Icon className="w-5 h-5 text-indigo-500" />
                                </div>
                                <div>
                                  <span className="thai-text font-bold text-slate-700 dark:text-slate-200 text-lg">{featureData.label}</span>
                                  <p className="text-sm text-slate-500 dark:text-slate-400 thai-text">{featureData.desc}</p>
                                </div>
                              </div>
                              <span className="font-black text-2xl text-transparent bg-clip-text bg-gradient-to-r from-indigo-500 to-purple-500">
                                {feat.importance}%
                              </span>
                            </div>
                            <div className="h-2 w-full bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden">
                              <motion.div 
                                initial={{ width: 0 }}
                                animate={{ width: `${feat.importance}%` }}
                                transition={{ duration: 1, delay: 0.5 + (index * 0.1) }}
                                className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full"
                              />
                            </div>
                          </div>
                        )
                      })}
                      {(!status?.feature_importance || status.feature_importance.length === 0) && (
                        <div className="text-center py-10 bg-slate-50 dark:bg-slate-900/50 rounded-2xl border border-slate-100 dark:border-slate-800">
                          <BrainCircuitIcon className="w-12 h-12 mx-auto text-slate-300 dark:text-slate-600 mb-3" />
                          <p className="text-slate-500 dark:text-slate-400 thai-text">ไม่มีข้อมูล Feature Importance จากโมเดลปัจจุบัน</p>
                        </div>
                      )}
                    </div>
                  </motion.div>

                </div>

                {/* Right Column (Controls & History) */}
                <div className="xl:col-span-4 space-y-8">
                  
                  {/* Control Center */}
                  <motion.div 
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.3 }}
                    className="bg-slate-900 rounded-[2rem] p-8 shadow-2xl border border-slate-800 relative overflow-hidden"
                  >
                    <div className="absolute -top-10 -right-10 w-40 h-40 bg-indigo-500/30 rounded-full blur-3xl pointer-events-none" />
                    
                    <h2 className="text-2xl font-bold text-white thai-text mb-3">สอน AI เพิ่มเติม</h2>
                    <p className="text-slate-400 thai-text text-sm leading-relaxed mb-6">
                      หากคุณเพิ่งตรวจสอบคู่สินค้าในหน้า Catalog เสร็จ สามารถกดปุ่มนี้เพื่อนำข้อมูลใหม่ไปเทรนให้โมเดลเก่งขึ้นทันที!
                    </p>
                    
                    <button 
                      onClick={handleRetrain}
                      disabled={retraining}
                      className={`w-full py-5 rounded-2xl flex flex-col items-center justify-center gap-2 transition-all duration-300 relative group
                        ${retraining 
                          ? 'bg-slate-800 text-slate-400 cursor-not-allowed border border-slate-700' 
                          : 'bg-gradient-to-b from-indigo-500 to-indigo-700 text-white hover:from-indigo-400 hover:to-indigo-600 hover:shadow-[0_0_30px_rgba(99,102,241,0.3)]'
                        }`}
                    >
                      {retraining ? (
                        <>
                          <RefreshCwIcon className="w-8 h-8 animate-spin" />
                          <span className="thai-text font-bold text-lg">กำลังอัปเดตโมเดล...</span>
                        </>
                      ) : (
                        <>
                          <BrainCircuitIcon className="w-8 h-8 group-hover:scale-110 transition-transform" />
                          <span className="thai-text font-bold text-lg">สอนอัปเดตความรู้ตอนนี้</span>
                        </>
                      )}
                    </button>
                  </motion.div>

                  {/* Brain Evolution Stats */}
                  <motion.div 
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.4 }}
                    className="bg-white dark:bg-slate-800 rounded-[2rem] p-6 shadow-xl border border-slate-200 dark:border-slate-700 grid grid-cols-2 gap-4"
                  >
                    <div className="bg-emerald-50 dark:bg-emerald-900/20 p-5 rounded-2xl border border-emerald-100 dark:border-emerald-800/30">
                      <TargetIcon className="w-6 h-6 text-emerald-600 dark:text-emerald-400 mb-2" />
                      <div className="text-sm text-emerald-700 dark:text-emerald-500 thai-text font-medium mb-1">ความแม่นยำสูง</div>
                      <div className="text-3xl font-black text-emerald-700 dark:text-emerald-400">{status?.accuracy || 0}%</div>
                    </div>
                    <div className="bg-blue-50 dark:bg-blue-900/20 p-5 rounded-2xl border border-blue-100 dark:border-blue-800/30">
                      <DatabaseIcon className="w-6 h-6 text-blue-600 dark:text-blue-400 mb-2" />
                      <div className="text-sm text-blue-700 dark:text-blue-500 thai-text font-medium mb-1">ประสบการณ์คู่</div>
                      <div className="text-3xl font-black text-blue-700 dark:text-blue-400">{status?.total_samples || 0}</div>
                    </div>
                    <div className="col-span-2 bg-purple-50 dark:bg-purple-900/20 p-5 rounded-2xl border border-purple-100 dark:border-purple-800/30 flex items-center justify-between">
                      <div>
                        <div className="text-sm text-purple-700 dark:text-purple-500 thai-text font-medium mb-1">ความมั่นใจในการปัดตก</div>
                        <div className="text-3xl font-black text-purple-700 dark:text-purple-400">{status?.average_confidence || 0}%</div>
                      </div>
                      <ActivityIcon className="w-10 h-10 text-purple-300 dark:text-purple-700 opacity-50" />
                    </div>
                  </motion.div>

                  {/* Training History Section */}
                  {history && history.history.length > 0 && (
                    <motion.div
                      initial={{ opacity: 0, y: 30 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.5 }}
                      className="bg-white dark:bg-slate-800 rounded-[2rem] p-6 shadow-xl border border-slate-200 dark:border-slate-700 flex flex-col h-fit"
                    >
                      <div className="flex items-center gap-3 mb-6">
                        <HistoryIcon className="w-6 h-6 text-slate-400" />
                        <h2 className="text-xl font-bold text-slate-800 dark:text-white thai-text">ประวัติวิวัฒนาการ</h2>
                      </div>

                      <div className="space-y-3">
                        {history.history.slice(0, 5).map((session, index) => {
                          const prev = history.history[index + 1]
                          const accuracyPct = Math.round(session.test_accuracy * 100)
                          const prevPct = prev ? Math.round(prev.test_accuracy * 100) : null
                          const trend = prevPct === null ? null : accuracyPct > prevPct ? 'up' : accuracyPct < prevPct ? 'down' : 'same'

                          return (
                            <div
                              key={session.id}
                              className={`flex items-center justify-between p-4 rounded-2xl border transition-all 
                                ${index === 0 
                                  ? 'bg-indigo-50 dark:bg-indigo-900/30 border-indigo-200 dark:border-indigo-800/50' 
                                  : 'bg-slate-50 dark:bg-slate-900/50 border-slate-100 dark:border-slate-800'}`}
                            >
                              <div>
                                <p className="font-bold text-slate-700 dark:text-slate-300 text-sm thai-text">
                                  {new Date(session.training_date).toLocaleDateString('th-TH', { month: 'short', day: 'numeric' })}
                                </p>
                                <p className="text-xs text-slate-500 dark:text-slate-500 thai-text mt-1">
                                  {session.total_samples} คู่
                                </p>
                              </div>

                              <div className="flex flex-col items-end gap-1">
                                <div className={`font-black text-lg
                                  ${accuracyPct >= 80 ? 'text-emerald-600 dark:text-emerald-400' :
                                    accuracyPct >= 60 ? 'text-amber-500' : 'text-red-500'}`}>
                                  {accuracyPct}%
                                </div>
                                {trend === 'up' && (
                                  <div className="flex items-center text-emerald-500 text-xs font-bold">
                                    <TrendingUpIcon className="w-3 h-3 mr-1" /> +{accuracyPct - (prevPct ?? 0)}%
                                  </div>
                                )}
                                {trend === 'down' && (
                                  <div className="flex items-center text-red-500 text-xs font-bold">
                                    <TrendingDownIcon className="w-3 h-3 mr-1" /> {accuracyPct - (prevPct ?? 0)}%
                                  </div>
                                )}
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    </motion.div>
                  )}
                </div>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}
