'use client'

import { useState, useEffect } from 'react'
import { 
  SparklesIcon, 
  BrainCircuitIcon, 
  Trash2Icon, 
  CheckCircle2Icon,
  SearchIcon,
  RefreshCcwIcon
} from 'lucide-react'
import { motion } from 'framer-motion'
import { supabase } from '@/utils/supabase'

interface KeywordRule {
  id: string
  code: string
  name: string
  keywords: string[]
  category_id: string
  match_type: string
  created_at: string
  taxonomy_nodes?: {
    name_th: string
  }
}

export default function AutoLearnTab() {
  const [rules, setRules] = useState<KeywordRule[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')

  useEffect(() => {
    fetchAutoLearnedRules()
  }, [])

  const fetchAutoLearnedRules = async () => {
    try {
      setIsLoading(true)
      const { data, error } = await supabase
        .from('keyword_rules')
        .select('*, taxonomy_nodes(name_th)')
        .eq('match_type', 'auto_learned')
        .order('created_at', { ascending: false })

      if (error) throw error
      setRules(data || [])
    } catch (error) {
      console.error('Error fetching rules:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const deleteRule = async (id: string) => {
    if (!confirm('คุณแน่ใจหรือไม่ที่จะลบ Keyword Rule นี้?')) return
    
    try {
      const { error } = await supabase
        .from('keyword_rules')
        .delete()
        .eq('id', id)

      if (error) throw error
      setRules(rules.filter(r => r.id !== id))
    } catch (error) {
      alert('เกิดข้อผิดพลาดในการลบ')
    }
  }

  const filteredRules = rules.filter(rule => 
    rule.keywords.some(k => k.toLowerCase().includes(searchTerm.toLowerCase())) ||
    rule.taxonomy_nodes?.name_th.includes(searchTerm)
  )

  return (
    <div className="h-full bg-gray-50 font-sans">
      <div className="flex-1 flex flex-col overflow-hidden relative">
        <main className="flex-1 overflow-x-hidden overflow-y-auto p-6 md:p-10 relative z-10">
          <div className="max-w-6xl mx-auto">
            {/* Header Section */}
            <div className="flex flex-col md:flex-row md:items-center justify-between mb-12 gap-6">
              <div>
                <div className="flex items-center gap-3 mb-4">
                   <div className="p-2 bg-indigo-600 rounded-xl shadow-lg shadow-indigo-100">
                      <SparklesIcon className="w-5 h-5 text-white" />
                   </div>
                   <span className="text-xs font-black text-indigo-600 uppercase tracking-[0.2em]">Autonomous Intelligence</span>
                </div>
                <h1 className="text-3xl md:text-4xl font-black text-slate-900 tracking-tight thai-text">
                  ระบบเรียนรู้อัตโนมัติ (Auto-learn)
                </h1>
                <p className="text-slate-500 mt-2 font-medium thai-text">
                  ตรวจสอบและจัดการคำสำคัญ (Keywords) ที่ AI เรียนรู้จากการยืนยันข้อมูลของคุณ
                </p>
              </div>

              <button 
                onClick={fetchAutoLearnedRules}
                className="flex items-center gap-2 px-6 py-3 bg-white border border-slate-200 rounded-2xl text-sm font-bold text-slate-600 hover:bg-slate-50 transition-all shadow-sm active:scale-95"
              >
                <RefreshCcwIcon className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
                รีเฟรชข้อมูล
              </button>
            </div>

            {/* Stats Mini cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
               <div className="premium-card p-6 bg-white border-white/80">
                  <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">Total Rules</p>
                  <p className="text-3xl font-black text-slate-900">{rules.length}</p>
               </div>
               <div className="premium-card p-6 bg-white border-white/80">
                  <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">Total Keywords</p>
                  <p className="text-3xl font-black text-indigo-600">
                    {rules.reduce((acc, curr) => acc + (curr.keywords?.length || 0), 0)}
                  </p>
               </div>
               <div className="premium-card p-6 bg-indigo-600 border-none shadow-xl shadow-indigo-100">
                  <p className="text-[10px] font-black text-indigo-200 uppercase tracking-widest mb-1">AI Status</p>
                  <div className="flex items-center gap-2 text-white">
                    <BrainCircuitIcon className="w-5 h-5 animate-pulse" />
                    <span className="text-lg font-black italic tracking-widest">Active & Learning</span>
                  </div>
               </div>
            </div>

            {/* Search Bar */}
            <div className="relative mb-8 group">
              <SearchIcon className="absolute left-6 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400 group-focus-within:text-indigo-500 transition-colors" />
              <input 
                type="text" 
                placeholder="ค้นหาตาม Keyword หรือหมวดหมู่..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-16 pr-6 py-5 bg-white border border-slate-200 rounded-[24px] text-slate-900 font-medium placeholder:text-slate-400 focus:outline-none focus:ring-4 focus:ring-indigo-500/10 focus:border-indigo-500 transition-all shadow-sm shadow-slate-100/50 thai-text"
              />
            </div>

            {/* Rules Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {filteredRules.map((rule, idx) => (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  key={rule.id}
                  className="premium-card p-8 bg-white border-white hover:shadow-xl transition-all group"
                >
                  <div className="flex justify-between items-start mb-6">
                    <div className="min-w-0">
                      <span className="px-3 py-1 bg-emerald-50 text-emerald-600 text-[10px] font-black rounded-full border border-emerald-100 uppercase tracking-widest">
                        Learned {new Date(rule.created_at).toLocaleDateString('th-TH')}
                      </span>
                      <h3 className="text-xl font-black text-slate-900 mt-3 tracking-tight truncate thai-text">
                        {rule.taxonomy_nodes?.name_th || 'ไม่ระบุหมวดหมู่'}
                      </h3>
                      <p className="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em] mt-1">{rule.code}</p>
                    </div>
                    <button 
                      onClick={() => deleteRule(rule.id)}
                      className="p-3 text-slate-300 hover:text-rose-500 hover:bg-rose-50 rounded-2xl transition-all opacity-0 group-hover:opacity-100"
                    >
                      <Trash2Icon className="w-5 h-5" />
                    </button>
                  </div>

                  <div className="flex flex-wrap gap-2">
                    {rule.keywords?.map((kw, i) => (
                      <span 
                        key={i}
                        className="px-4 py-2 bg-slate-50 text-slate-700 text-xs font-bold rounded-xl border border-slate-100 flex items-center gap-2 group/kw hover:bg-white hover:border-indigo-200 transition-all thai-text cursor-default"
                      >
                        <CheckCircle2Icon className="w-3 h-3 text-indigo-400" />
                        {kw}
                      </span>
                    ))}
                  </div>
                </motion.div>
              ))}

              {filteredRules.length === 0 && !isLoading && (
                <div className="col-span-full py-20 flex flex-col items-center justify-center bg-white rounded-[40px] border border-dashed border-slate-200">
                  <div className="w-20 h-20 bg-slate-50 rounded-full flex items-center justify-center mb-6 text-slate-300">
                    <SearchIcon className="w-10 h-10" />
                  </div>
                  <p className="text-slate-400 font-bold uppercase tracking-widest text-xs">No auto-learned keywords found</p>
                  <p className="text-slate-300 mt-2 thai-text">เริ่ม Approve สินค้าเพื่อสอน AI ของคุณ!</p>
                </div>
              )}
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
