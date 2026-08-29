'use client'

import { useState, useEffect, useCallback } from 'react'
import { 
  CheckCircle, 
  XCircle, 
  Search, 
  Tag, 
  Layers, 
  AlertCircle,
  Database,
  ArrowRight,
  Filter,
  RefreshCcw,
  Save,
  Trash2
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { toast } from 'react-hot-toast'
import { supabase } from '@/utils/supabase'


interface Product {
  id: string
  name_th: string
  status: string
  confidence_score: number
  category_id?: string
  metadata: any
  created_at: string
}

export default function VerifyTab() {
  const [activeTab, setActiveTab] = useState<'dedup' | 'category' | 'approved'>('dedup')
  const [products, setProducts] = useState<Product[]>([])
  const [loading, setLoading] = useState(true)
  const [taxonomy, setTaxonomy] = useState<any[]>([])
  const [stats, setStats] = useState({ dedup: 0, category: 0, approved: 0 })
  const [busyId, setBusyId] = useState<string | null>(null)

  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      // 1. Fetch Stats
      const { count: dedupCount } = await supabase
        .from('products')
        .select('*', { count: 'exact', head: true })
        .eq('status', 'pending_review_dedup')
      
      const { count: catCount } = await supabase
        .from('products')
        .select('*', { count: 'exact', head: true })
        .eq('status', 'pending_review_category')

      const { count: approvedCount } = await supabase
        .from('products')
        .select('*', { count: 'exact', head: true })
        .eq('status', 'approved')
      
      setStats({ 
        dedup: dedupCount || 0, 
        category: catCount || 0,
        approved: approvedCount || 0
      })

      // 2. Fetch Active Tab Products
      const statusMap = {
        'dedup': 'pending_review_dedup',
        'category': 'pending_review_category',
        'approved': 'approved'
      }
      
      const { data } = await supabase
        .from('products')
        .select('*')
        .eq('status', statusMap[activeTab])
        .order('created_at', { ascending: false })
        .limit(20)
      
      setProducts(data || [])
    } catch (err) {
      console.error('Error fetching data:', err)
    } finally {
      setLoading(false)
    }
  }, [activeTab])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  // แยกออกมาเป็น effect ของตัวเอง — เดิมอยู่ใน fetchData ที่มี taxonomy.length เป็น dependency
  // พอโหลดหมวดหมู่เสร็จ dependency ก็เปลี่ยน ทำให้โหลดรายการใหม่ทั้งชุดซ้ำอีกรอบทุกครั้งที่เข้าหน้า
  useEffect(() => {
    supabase
      .from('taxonomy_nodes')
      .select('id, name_th')
      .then(({ data }) => setTaxonomy(data || []))
  }, [])

  /**
   * ทุกคำตัดสินต้องผ่าน /api/verify ที่ใช้ service role
   * เขียนตรงด้วย anon key ไม่ได้ — RLS กรองแถวทิ้งก่อน แล้วคืน 200 พร้อมของว่าง
   * หน้าเว็บจึงเคยขึ้นว่าเรียบร้อยทั้งที่ไม่มีอะไรถูกบันทึกเลย
   */
  const submitDecision = async (
    id: string,
    action: 'keep' | 'discard' | 'confirm_category',
    categoryId?: string
  ) => {
    setBusyId(id)
    try {
      const response = await fetch('/api/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ product_id: id, action, category_id: categoryId }),
      })
      const payload = await response.json()
      if (!response.ok || !payload.success) {
        throw new Error(payload.error || 'บันทึกไม่สำเร็จ')
      }

      // เอารายการที่ตัดสินแล้วออกเมื่อ route ยืนยันว่าบันทึกจริง ไม่ใช่ตัดออกไปก่อน
      setProducts(current => current.filter(p => p.id !== id))
      setStats(current => ({
        ...current,
        [activeTab]: Math.max(current[activeTab] - 1, 0),
        approved: action === 'confirm_category' ? current.approved + 1 : current.approved,
      }))
      toast.success('บันทึกเรียบร้อย')
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'บันทึกไม่สำเร็จ')
    } finally {
      setBusyId(null)
    }
  }

  const handleDedupDecision = (id: string, decision: 'keep' | 'discard') =>
    submitDecision(id, decision)

  const handleCategoryConfirm = (id: string, categoryId: string) => {
    if (!categoryId) {
      toast.error('กรุณาเลือกหมวดหมู่ก่อนยืนยัน')
      return
    }
    return submitDecision(id, 'confirm_category', categoryId)
  }

  return (
    <div className="h-full bg-[#F8FAFC]">
      <div className="flex-1 flex flex-col overflow-hidden">
        <main className="flex-1 overflow-y-auto p-8">
          <div className="max-w-6xl mx-auto">
            <div className="flex justify-between items-end mb-8">
              <div>
                <h1 className="text-3xl font-black text-slate-900 tracking-tight flex items-center gap-3">
                  <Database className="w-8 h-8 text-indigo-600" />
                  Verification Center
                </h1>
                <p className="text-slate-500 font-medium mt-1">ศูนย์ตรวจสอบและยืนยันความถูกต้องของข้อมูล (Human-in-the-loop)</p>
              </div>
              <button 
                onClick={fetchData}
                className="p-3 bg-white rounded-2xl border border-slate-200 shadow-sm hover:bg-slate-50 transition-colors"
              >
                <RefreshCcw className={`w-5 h-5 text-slate-600 ${loading ? 'animate-spin' : ''}`} />
              </button>
            </div>

            {/* Main Tabs */}
            <div className="flex gap-4 mb-8 p-1.5 bg-slate-200/50 rounded-[24px] w-fit">
              <button 
                onClick={() => setActiveTab('dedup')}
                className={`px-8 py-3 rounded-[20px] text-sm font-black transition-all flex items-center gap-2 ${
                  activeTab === 'dedup' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'
                }`}
              >
                <Layers className="w-4 h-4" />
                ด่าน 1: ตรวจของซ้ำ ({stats.dedup})
              </button>
              <button
                onClick={() => setActiveTab('category')}
                data-testid="tab-verify-category"
                className={`px-8 py-3 rounded-[20px] text-sm font-black transition-all flex items-center gap-2 ${
                  activeTab === 'category' ? 'bg-white text-amber-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'
                }`}
              >
                <Tag className="w-4 h-4" />
                ด่าน 2: ตรวจหมวดหมู่ ({stats.category})
              </button>
              <button 
                onClick={() => setActiveTab('approved')}
                className={`px-8 py-3 rounded-[20px] text-sm font-black transition-all flex items-center gap-2 ${
                  activeTab === 'approved' ? 'bg-white text-emerald-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'
                }`}
              >
                <CheckCircle className="w-4 h-4" />
                ผ่านฉลุย ({stats.approved})
              </button>
            </div>

            {loading ? (
              <div className="h-64 flex items-center justify-center">
                <RefreshCcw className="w-10 h-10 text-indigo-200 animate-spin" />
              </div>
            ) : (
              <div className="space-y-6">
                {products.length === 0 ? (
                  <div className="bg-white rounded-[40px] p-20 text-center border-2 border-dashed border-slate-200">
                    <div className="w-20 h-20 bg-slate-50 rounded-full flex items-center justify-center mx-auto mb-6">
                      <CheckCircle className="w-10 h-10 text-slate-200" />
                    </div>
                    <h3 className="text-xl font-bold text-slate-400">ไม่มีรายการค้างตรวจสอบในด่านนี้</h3>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 gap-6">
                    {activeTab === 'dedup' ? (
                      products.map((p) => (
                        <div key={p.id} className="bg-white rounded-[32px] p-8 shadow-sm border border-slate-100 hover:border-indigo-200 transition-all group">
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <div className="flex items-center gap-2 mb-4">
                                <span className="px-3 py-1 bg-amber-50 text-amber-600 text-[10px] font-black uppercase tracking-widest rounded-lg">Potential Duplicate</span>
                                <span className="text-slate-300 text-xs">•</span>
                                <span className="text-slate-400 text-xs font-bold">Match Score: {(p.metadata?.similarity_score * 100).toFixed(1)}%</span>
                              </div>
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
                                <div>
                                  <p className="text-xs font-black text-slate-400 uppercase tracking-widest mb-2">ของใหม่ที่พบ</p>
                                  <p className="text-xl font-black text-slate-900 leading-relaxed">{p.name_th}</p>
                                  <p className="text-sm text-slate-400 mt-2 italic font-medium">Cleaned: {p.metadata?.clean_name}</p>
                                </div>
                                <div className="bg-slate-50 p-6 rounded-2xl border border-slate-100">
                                  <p className="text-xs font-black text-slate-400 uppercase tracking-widest mb-2">คล้ายกับของเดิมชื่อ</p>
                                  <p className="text-lg font-bold text-slate-600">{p.metadata?.duplicate_of}</p>
                                </div>
                              </div>
                            </div>
                            <div className="flex flex-col gap-3 ml-8">
                              <button
                                onClick={() => handleDedupDecision(p.id, 'discard')}
                                disabled={busyId === p.id}
                                data-testid="discard-btn"
                                className="p-4 bg-rose-50 text-rose-600 rounded-2xl hover:bg-rose-600 hover:text-white transition-all shadow-sm disabled:opacity-40"
                                title="ซ้ำจริง - ลบทิ้ง"
                              >
                                <Trash2 className="w-6 h-6" />
                              </button>
                              <button
                                onClick={() => handleDedupDecision(p.id, 'keep')}
                                disabled={busyId === p.id}
                                data-testid="keep-btn"
                                className="p-4 bg-emerald-50 text-emerald-600 rounded-2xl hover:bg-emerald-600 hover:text-white transition-all shadow-sm disabled:opacity-40"
                                title="ไม่ซ้ำ - ไปด่านต่อไป"
                              >
                                <CheckCircle className="w-6 h-6" />
                              </button>
                            </div>
                          </div>
                        </div>
                      ))
                    ) : (
                      products.map((p) => (
                        <div key={p.id} className="bg-white rounded-[32px] p-8 shadow-sm border border-slate-100 hover:border-emerald-200 transition-all group">
                          <div className="flex items-center justify-between">
                            <div className="flex-1">
                              <div className="flex items-center gap-2 mb-4">
                                <span className="px-3 py-1 bg-indigo-50 text-indigo-600 text-[10px] font-black uppercase tracking-widest rounded-lg">New Unique Product</span>
                                <span className="text-slate-300 text-xs">•</span>
                                <span className="text-slate-400 text-xs font-bold">AI Confidence: {(p.confidence_score * 100).toFixed(0)}%</span>
                              </div>
                              <p className="text-2xl font-black text-slate-900 mb-2">{p.name_th}</p>
                              <p className="text-sm text-slate-400 font-medium mb-6 italic">Cleaned: {p.metadata?.clean_name}</p>
                              
                              <div className="flex flex-wrap items-center gap-4">
                                <div className="flex items-center gap-3 bg-slate-50 px-6 py-3 rounded-2xl border border-slate-100">
                                  <Tag className="w-4 h-4 text-emerald-500" />
                                  <span className="text-sm font-black text-slate-700">AI แนะนำ: {p.metadata?.suggested_category}</span>
                                </div>
                                <ArrowRight className="w-4 h-4 text-slate-300" />
                                <select
                                  data-testid="category-select"
                                  disabled={busyId === p.id}
                                  className="px-6 py-3 bg-white border border-slate-200 rounded-2xl text-sm font-bold text-slate-700 focus:ring-2 focus:ring-emerald-500 outline-none disabled:opacity-40"
                                  defaultValue={p.category_id}
                                  onChange={(e) => handleCategoryConfirm(p.id, e.target.value)}
                                >
                                  <option value="">เลือกหมวดหมู่ที่ถูกต้อง...</option>
                                  {taxonomy.map(t => (
                                    <option key={t.id} value={t.id}>{t.name_th}</option>
                                  ))}
                                </select>
                              </div>
                            </div>
                            <div className="ml-8">
                              <button
                                onClick={() => handleCategoryConfirm(p.id, p.category_id || '')}
                                disabled={busyId === p.id}
                                data-testid="confirm-category-btn"
                                className="px-8 py-4 bg-emerald-600 text-white rounded-[20px] font-black text-sm shadow-lg shadow-emerald-200 hover:scale-105 active:scale-95 transition-all flex items-center gap-3 disabled:opacity-40"
                              >
                                <Save className="w-5 h-5" />
                                ยืนยันหมวดนี้
                              </button>
                            </div>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}
