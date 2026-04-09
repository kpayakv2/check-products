'use client'

import { useState, useCallback, useMemo } from 'react'
import { 
  Upload, 
  FileText, 
  CheckCircle, 
  XCircle, 
  AlertCircle, 
  Download, 
  Tag, 
  ArrowRight, 
  LayersIcon, 
  ZapIcon, 
  SearchIcon, 
  RefreshCcwIcon,
  ActivityIcon,
  ChevronRightIcon,
  SparklesIcon,
  ScaleIcon
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { supabase } from '@/utils/supabase'
import Sidebar from '@/components/Layout/Sidebar'
import Header from '@/components/Layout/Header'

interface DeduplicationResult {
  id: string
  newProduct: string
  oldProduct: string
  similarity: number
  confidence: number
  mlPrediction: 'similar' | 'different'
  status: 'pending' | 'approved' | 'rejected'
  reason: string
}

interface ProcessedProduct {
  id: string
  name_th: string
  category?: string
  categoryId?: string
  confidence?: number
  status: 'unique' | 'categorized' | 'ready'
  attributes?: Record<string, any>
}

interface WorkflowStep {
  id: number
  name: string
  description: string
  status: 'pending' | 'processing' | 'completed' | 'error'
  count?: number
}

export default function ProductDeduplication() {
  const [oldFile, setOldFile] = useState<File | null>(null)
  const [newFile, setNewFile] = useState<File | null>(null)
  const [results, setResults] = useState<DeduplicationResult[]>([])
  const [uniqueProducts, setUniqueProducts] = useState<ProcessedProduct[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [currentIndex, setCurrentIndex] = useState(0)
  const [currentStep, setCurrentStep] = useState(1)
  
  const workflowSteps: WorkflowStep[] = [
    { id: 1, name: 'Data Ingestion', description: 'Upload Source Files', status: 'pending' },
    { id: 2, name: 'AI Conflict Check', description: 'Neural Deduplication', status: 'pending' },
    { id: 3, name: 'Human Audit', description: 'Manual Verification', status: 'pending' },
    { id: 4, name: 'Auto-Classification', description: 'Taxonomy Alignment', status: 'pending' },
    { id: 5, name: 'Final Build', description: 'POS Ready Export', status: 'pending' }
  ]

  const handleFileUpload = useCallback(async (file: File, type: 'old' | 'new') => {
    if (type === 'old') setOldFile(file)
    else setNewFile(file)
  }, [])

  const analyzeProducts = useCallback(async () => {
    if (!oldFile || !newFile) return
    setLoading(true)
    setError(null)
    setCurrentStep(2)
    try {
      const { data: oldUpload } = await supabase.storage.from('uploads').upload(`old_${Date.now()}.csv`, oldFile)
      const { data: newUpload } = await supabase.storage.from('uploads').upload(`new_${Date.now()}.csv`, newFile)

      const { data, error: funcError } = await supabase.functions.invoke('product-deduplication', {
        body: { oldProductsPath: oldUpload?.path, newProductsPath: newUpload?.path, threshold: 0.75 }
      })

      if (funcError) throw funcError
      setResults(data.results || [])
      
      if (data.uniqueProducts?.length > 0) {
        setUniqueProducts(data.uniqueProducts.map((name_th: string, i: number) => ({
          id: `unique_${i + 1}`, name_th, status: 'unique'
        })))
      }
      
      setCurrentStep(data.results?.length > 0 ? 3 : 4)
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Analysis failed')
    } finally {
      setLoading(false)
    }
  }, [oldFile, newFile])

  const categorizeProducts = useCallback(async () => {
    setLoading(true)
    setCurrentStep(4)
    try {
      const approvedFromReview = results.filter(r => r.status === 'approved').map(r => ({ id: r.id, name_th: r.newProduct, status: 'unique' as const }))
      const allUniqueProducts = [...uniqueProducts, ...approvedFromReview]
      const categorizedProducts: ProcessedProduct[] = []
      
      for (const product of allUniqueProducts) {
        const { data } = await supabase.functions.invoke('category-suggestions', { body: { text: product.name_th } })
        const sug = data.suggestions?.[0]
        categorizedProducts.push({ ...product, category: sug?.categoryName || 'Unmapped', confidence: sug?.confidence || 0, status: 'categorized' })
      }
      setUniqueProducts(categorizedProducts)
      setCurrentStep(5)
    } finally {
      setLoading(false)
    }
  }, [uniqueProducts, results])

  const exportToCSV = useCallback(() => {
    const readyProducts = uniqueProducts.filter(p => p.status === 'categorized')
    const headers = ['ID', 'Name_TH', 'Category', 'Confidence']
    const csvContent = headers.join(',') + '\n' + readyProducts.map(p => `${p.id},"${p.name_th}","${p.category}",${p.confidence}`).join('\n')
    const blob = new Blob(['\uFEFF' + csvContent], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `pos_export_${Date.now()}.csv`
    link.click()
    setUniqueProducts(prev => prev.map(p => p.status === 'categorized' ? { ...p, status: 'ready' } : p))
  }, [uniqueProducts])

  const handleReview = useCallback(async (decision: 'similar' | 'different' | 'duplicate') => {
    const currentResult = results[currentIndex]
    if (!currentResult) return
    const updatedResults = [...results]
    updatedResults[currentIndex].status = decision === 'duplicate' ? 'rejected' : 'approved'
    setResults(updatedResults)
    if (currentIndex < results.length - 1) setCurrentIndex(currentIndex + 1)
  }, [results, currentIndex])

  const progress = results.length > 0 ? ((currentIndex + 1) / results.length) * 100 : 0
  const currentResult = results[currentIndex]

  return (
    <div className="flex h-screen bg-gray-50 font-sans">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-x-hidden overflow-y-auto p-8 relative">
          {/* Decorative Background Elements */}
          <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-indigo-50/40 rounded-full blur-[120px] -mr-64 -mt-64 pointer-events-none" />
          <div className="absolute bottom-10 left-10 w-[400px] h-[400px] bg-emerald-50/40 rounded-full blur-[120px] pointer-events-none" />

          <div className="max-w-7xl mx-auto relative z-10">
            {/* Page Header */}
            <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between mb-12 gap-6">
              <div>
                <h1 className="text-3xl font-black text-slate-900 tracking-tight uppercase thai-text flex items-center gap-4">
                  Deduplication Matrix
                </h1>
                <p className="text-slate-500 mt-2 font-medium thai-text leading-relaxed max-w-2xl">
                  ระบบตรวจสอบและป้องกันข้อมูลสินค้าซ้ำซ้อนด้วย AI เพื่อความสะอาดของฐานข้อมูล (Data Deduplication) และการสมานข้อมูลสินค้า (Data Harmonization)
                </p>
              </div>
              
              <div className="flex gap-4 p-3 bg-white/60 backdrop-blur-md rounded-[32px] border border-white shadow-sm px-8">
                 <div className="flex items-center gap-3 pr-6 border-r border-slate-100">
                    <div className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse" />
                    <span className="text-2xl font-black text-slate-900 leading-none">{uniqueProducts.length}</span>
                    <span className="text-xs font-black text-slate-400 uppercase tracking-widest leading-none">Processed<br/>Units</span>
                 </div>
                 <div className="flex items-center gap-3">
                    <div className="w-2 h-2 rounded-full bg-emerald-500" />
                    <span className="text-2xl font-black text-slate-900 leading-none">
                      {uniqueProducts.filter(p => p.status === 'ready').length}
                    </span>
                    <span className="text-xs font-black text-slate-400 uppercase tracking-widest leading-none">Success<br/>Syncs</span>
                 </div>
              </div>
            </div>

            {/* Workflow Stepper */}
            <div className="premium-card p-10 mb-12 bg-white/40 border-white/80 overflow-hidden">
               <div className="relative flex items-center justify-between">
                  <div className="absolute top-6 left-0 right-0 h-1 bg-slate-100 rounded-full overflow-hidden">
                     <motion.div 
                        initial={{ width: 0 }}
                        animate={{ width: `${(currentStep - 1) * 25}%` }}
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

            {/* Main Action Zone */}
            <div className="grid grid-cols-1 lg:grid-cols-1 gap-12 mb-16">
              {currentStep === 1 && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                  {[
                    { file: oldFile, set: setOldFile, title: 'Master Repository', desc: 'Baseline Data', type: 'old', icon: SearchIcon, color: 'indigo' },
                    { file: newFile, set: setNewFile, title: 'Input Stream', desc: 'Candidate Batch', type: 'new', icon: ActivityIcon, color: 'emerald' }
                  ].map((target, i) => (
                    <div key={i} className={`premium-card p-1 pb-1 border-2 border-dashed transition-all duration-500 ${
                       target.file ? `bg-${target.color}-50/30 border-${target.color}-200` : 'bg-white/40 border-slate-100 hover:border-indigo-200'
                    }`}>
                       <div className="p-12 text-center flex flex-col items-center justify-center h-full">
                          <div className={`w-20 h-20 rounded-3xl flex items-center justify-center mb-8 transition-all duration-500 ${
                             target.file ? `bg-${target.color}-600 text-white shadow-2xl shadow-${target.color}-200 scale-110 rotate-3` : 'bg-slate-50 text-slate-200 shadow-inner'
                          }`}>
                             <target.icon className="w-10 h-10" />
                          </div>
                          <h3 className="text-xl font-black text-slate-900 mb-3 uppercase tracking-tight thai-text">{target.title}</h3>
                          <p className="text-xs font-black text-slate-400 uppercase tracking-widest mb-10">{target.desc}</p>
                          
                          <input type="file" onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0], target.type as any)} className="hidden" id={`file-${target.type}`} />
                          {!target.file ? (
                             <label htmlFor={`file-${target.type}`} className="btn-premium-secondary py-4 px-10 cursor-pointer">Select Asset</label>
                          ) : (
                             <div className="space-y-4">
                                <p className="text-sm font-black text-slate-800 bg-white px-6 py-2 rounded-xl border border-slate-100 shadow-sm">{target.file.name}</p>
                                <button onClick={() => target.set(null)} className="text-xs font-black text-rose-500 uppercase tracking-widest hover:underline">Revoke Source</button>
                             </div>
                          )}
                       </div>
                    </div>
                  ))}
                  
                  <div className="md:col-span-2 flex justify-center pt-8">
                     <button onClick={analyzeProducts} disabled={!oldFile || !newFile || loading} className="btn-premium px-20 py-6 text-xl group h-auto">
                        {loading ? <RefreshCcwIcon className="animate-spin w-6 h-6 mr-3" /> : <ZapIcon className="w-6 h-6 mr-3 group-hover:animate-pulse" />}
                        <span className="thai-text uppercase tracking-widest font-black">Invoke Processor</span>
                     </button>
                  </div>
                </div>
              )}

              {currentStep === 3 && currentResult && (
                <div className="space-y-12 animate-in fade-in slide-in-from-bottom-12 duration-700">
                  <div className="flex items-center justify-between">
                     <div>
                        <h3 className="text-2xl font-black text-slate-900 tracking-tight uppercase">Audit Required (Conflicts Identified)</h3>
                        <p className="text-indigo-500 text-xs font-black uppercase tracking-[0.2em] mt-1">Manual Resolution Protocol</p>
                     </div>
                     <div className="px-6 py-2 bg-indigo-500 rounded-2xl text-white font-black text-sm shadow-lg shadow-indigo-100 tracking-tighter italic">
                        {currentIndex + 1} / {results.length}
                     </div>
                  </div>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 relative">
                     <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-20 hidden lg:flex w-20 h-20 rounded-full bg-white shadow-2xl border border-slate-50 items-center justify-center">
                        <ScaleIcon className="w-8 h-8 text-indigo-500 animate-pulse" />
                     </div>

                     {[
                       { type: 'Incoming', name_th: currentResult.newProduct, color: 'indigo', icon: ActivityIcon },
                       { type: 'Master', name_th: currentResult.oldProduct, color: 'slate', icon: SearchIcon }
                     ].map((item, i) => (
                       <div key={i} className={`premium-card p-0 overflow-hidden bg-white/80 border-${i === 0 ? 'indigo-100' : 'slate-100'}`}>
                          <div className={`px-6 py-3 bg-${i === 0 ? 'indigo-600' : 'slate-900'} text-white flex justify-between items-center`}>
                             <span className="text-xs font-black uppercase tracking-[0.2em]">{item.type} Identity</span>
                             <item.icon className="w-4 h-4 opacity-50" />
                          </div>
                          <div className="p-12">
                             <p className="text-2xl font-black text-slate-900 thai-text leading-relaxed tracking-tight min-h-[6rem]">
                                {item.name_th}
                             </p>
                          </div>
                       </div>
                     ))}
                  </div>

                  <div className="bg-slate-900 p-12 rounded-[56px] text-center shadow-2xl relative overflow-hidden">
                     <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-500/10 rounded-full blur-[80px] -mr-32 -mt-32" />
                     
                     <div className="inline-flex gap-2 items-center bg-white/5 border border-white/10 px-6 py-2 rounded-full mb-10">
                        <span className="text-xs font-black text-indigo-400 uppercase tracking-widest">Neural Score:</span>
                        <span className="text-base font-black text-white italic tracking-widest">{(currentResult.similarity * 100).toFixed(1)}% Match</span>
                     </div>

                     <h4 className="text-xl font-black text-white thai-text mb-12 uppercase tracking-tight">เลือกดำเนินการเพื่อสมานข้อมูล (Data Harmonization Decision)</h4>

                     <div className="flex flex-wrap justify-center gap-10">
                        {[
                          { id: 'different', label: 'Adopt New', desc: 'Distinct Entry', icon: CheckCircle, color: 'emerald' },
                          { id: 'duplicate', label: 'Is Duplicate', desc: 'Reject Input', icon: XCircle, color: 'rose' },
                          { id: 'similar', label: 'Review Later', desc: 'Escalate Logic', icon: AlertCircle, color: 'amber' }
                        ].map((btn) => (
                           <button key={btn.id} onClick={() => handleReview(btn.id as any)} className="group flex flex-col items-center gap-4">
                              <div className={`w-20 h-20 rounded-3xl bg-white/5 border border-white/10 flex items-center justify-center transition-all duration-500 group-hover:bg-${btn.color}-500 group-hover:scale-110 group-hover:rotate-3 group-hover:shadow-2xl group-hover:shadow-${btn.color}-500/50`}>
                                 <btn.icon className={`w-8 h-8 text-${btn.color}-500 group-hover:text-white transition-colors`} />
                              </div>
                              <div className="text-center">
                                 <p className="text-xs font-black text-white uppercase tracking-widest">{btn.label}</p>
                                 <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mt-1 opacity-0 group-hover:opacity-100 transition-opacity">{btn.desc}</p>
                              </div>
                           </button>
                        ))}
                     </div>
                  </div>
                </div>
              )}

              {(currentStep === 4 || currentStep === 5) && (
                <div className="space-y-10 animate-in fade-in zoom-in duration-500">
                  <div className="flex items-baseline justify-between mb-2">
                    <div>
                      <h3 className="text-3xl font-black text-slate-900 tracking-tight uppercase">Harmonized Inventory</h3>
                      <p className="text-emerald-500 text-xs font-black uppercase tracking-[0.2em] mt-1">Ready for Sync Integration</p>
                    </div>
                    {currentStep === 4 ? (
                      <button onClick={categorizeProducts} className="btn-premium group px-12 py-5 h-auto">
                        <SparklesIcon className="w-5 h-5 mr-3 group-hover:rotate-12 transition-transform" />
                        <span className="thai-text uppercase font-black">Auto-Align Categories</span>
                      </button>
                    ) : (
                      <button onClick={exportToCSV} className="btn-premium bg-emerald-600 hover:bg-emerald-700 px-12 py-5 h-auto flex items-center shadow-emerald-500/30">
                        <Download className="w-5 h-5 mr-3" />
                        <span className="thai-text uppercase font-black">Export POS Manifest</span>
                      </button>
                    )}
                  </div>

                  <div className="premium-card overflow-hidden bg-white/60 border-white shadow-xl">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b border-slate-100 bg-slate-50/50">
                          <th className="px-10 py-6 text-left text-xs font-black text-slate-400 uppercase tracking-widest">Asset Identifier</th>
                          <th className="px-10 py-6 text-left text-xs font-black text-slate-400 uppercase tracking-widest">Classification</th>
                          <th className="px-10 py-6 text-left text-xs font-black text-slate-400 uppercase tracking-widest">Confidence Index</th>
                          <th className="px-10 py-6 text-left text-xs font-black text-slate-400 uppercase tracking-widest">Protocol Status</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {uniqueProducts.filter(p => p.status === 'categorized' || p.status === 'ready').map((p, idx) => (
                           <motion.tr 
                              key={p.id}
                              initial={{ opacity: 0, x: -10 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: idx * 0.05 }}
                              className="group hover:bg-indigo-50/20 transition-colors"
                           >
                              <td className="px-10 py-6 text-sm font-black text-slate-700 thai-text tracking-tight">{p.name_th}</td>
                              <td className="px-10 py-6">
                                 <span className="flex items-center gap-2 text-[11px] font-black tracking-widest text-indigo-600 uppercase">
                                    <Tag className="w-3.5 h-3.5" />
                                    {p.category}
                                 </span>
                              </td>
                              <td className="px-10 py-6">
                                 <div className="flex items-center gap-4">
                                    <div className="w-24 h-1.5 bg-slate-100 rounded-full overflow-hidden shadow-inner flex-shrink-0">
                                       <motion.div initial={{ width: 0 }} animate={{ width: `${(p.confidence || 0) * 100}%` }} className={`h-full ${p.confidence! > 0.8 ? 'bg-emerald-500' : p.confidence! > 0.5 ? 'bg-amber-400' : 'bg-rose-500'}`} />
                                    </div>
                                    <span className="text-[10px] font-black text-slate-400">{((p.confidence || 0) * 100).toFixed(0)}%</span>
                                 </div>
                              </td>
                              <td className="px-10 py-6">
                                 <div className={`inline-flex items-center px-4 py-1.5 rounded-xl text-[10px] font-black uppercase tracking-widest border ${
                                    p.status === 'ready' ? 'bg-emerald-50 text-emerald-600 border-emerald-100' : 'bg-indigo-50 text-indigo-600 border-indigo-100'
                                 }`}>
                                    {p.status === 'ready' ? 'Synchronized' : 'Ready'}
                                 </div>
                              </td>
                           </motion.tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
