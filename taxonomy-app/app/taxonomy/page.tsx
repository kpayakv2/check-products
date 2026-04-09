'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { toast } from 'react-hot-toast'
import Sidebar from '@/components/Layout/Sidebar'
import Header from '@/components/Layout/Header'
import { TaxonomyNode, DatabaseService } from '@/utils/supabase'
import { 
  PlusIcon, 
  EditIcon, 
  TrashIcon, 
  SaveIcon, 
  XIcon,
  FolderIcon,
  InfoIcon,
  ChevronRightIcon,
  LayersIcon,
  ArrowRightIcon,
  FolderTreeIcon,
  SparklesIcon,
  SearchIcon,
  ActivityIcon
} from 'lucide-react'

export default function TaxonomyManager() {
  const [categories, setCategories] = useState<TaxonomyNode[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [showForm, setShowForm] = useState(false)
  const [editingCategory, setEditingCategory] = useState<TaxonomyNode | null>(null)
  const [formData, setFormData] = useState({
    name_th: '',
    name_en: '',
    description: '',
    parent_id: '',
    keywords: [] as string[]
  })

  useEffect(() => {
    loadCategories()
  }, [])

  const loadCategories = async () => {
    try {
      setIsLoading(true)
      const taxonomyData = await DatabaseService.getTaxonomyTree()
      setCategories(taxonomyData || [])
    } catch (error) {
      console.error('Error loading categories:', error)
      toast.error('เกิดข้อผิดพลาดในการโหลดข้อมูล')
      setCategories([])
    } finally {
      setIsLoading(false)
    }
  }

  const handleCategoryEdit = (category: TaxonomyNode) => {
    setEditingCategory(category)
    setFormData({
      name_th: category.name_th,
      name_en: category.name_en || '',
      description: category.description || '',
      parent_id: category.parent_id || '',
      keywords: category.keywords || []
    })
    setShowForm(true)
  }

  const handleCategoryDelete = async (category: TaxonomyNode) => {
    if (!confirm(`คุณต้องการลบหมวดหมู่ "${category.name_th}" หรือไม่?`)) {
      return
    }

    try {
      await DatabaseService.deleteTaxonomyNode(category.id)
      toast.success('ลบหมวดหมู่เรียบร้อยแล้ว')
      loadCategories()
    } catch (error) {
      console.error('Error deleting category:', error)
      toast.error('เกิดข้อผิดพลาดในการลบหมวดหมู่')
    }
  }

  const handleFormSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!formData.name_th.trim()) {
      toast.error('กรุณาใส่ชื่อหมวดหมู่ภาษาไทย')
      return
    }

    try {
      if (editingCategory) {
        await DatabaseService.updateTaxonomyNode(editingCategory.id, formData)
        toast.success('แก้ไขหมวดหมู่เรียบร้อยแล้ว')
      } else {
        // Generate a unique code for the new category
        const uniqueCode = `CAT-${Date.now().toString().slice(-6)}${Math.random().toString(36).substring(2, 5).toUpperCase()}`
        
        await DatabaseService.createTaxonomyNode({
          ...formData,
          code: uniqueCode,
          level: 0,
          sort_order: 0,
          is_active: true
        })
        toast.success('เพิ่มหมวดหมู่เรียบร้อยแล้ว')
      }
      
      setShowForm(false)
      setEditingCategory(null)
      setFormData({
        name_th: '',
        name_en: '',
        description: '',
        parent_id: '',
        keywords: []
      })
      loadCategories()
    } catch (error) {
      console.error('Error saving category:', error)
      toast.error('เกิดข้อผิดพลาดในการบันทึกหมวดหมู่')
    }
  }

  const renderCategoryTree = (nodes: TaxonomyNode[], level = 0) => {
    return nodes.map((node, idx) => (
      <motion.div
        key={node.id}
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: idx * 0.05 }}
        className={`mb-6 ${level > 0 ? 'ml-12 border-l-2 border-indigo-50/50 pl-10 relative' : ''}`}
      >
        {level > 0 && (
          <div className="absolute left-0 top-10 w-10 h-0.5 bg-indigo-50/50" />
        )}
        
        <div className="premium-card group hover:bg-white/95 p-6 border-white/80 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-50/20 rounded-full blur-2xl group-hover:bg-indigo-100/30 transition-colors" />
          
          <div className="flex items-center justify-between relative z-10">
            <div className="flex items-center space-x-6">
              <div className={`w-14 h-14 rounded-2xl bg-gradient-to-br from-indigo-500 to-indigo-600 flex items-center justify-center text-white shadow-xl shadow-indigo-100 transition-all duration-500 group-hover:scale-110 group-hover:rotate-3`}>
                <FolderTreeIcon className="h-6 w-6" />
              </div>
              <div>
                <div className="flex items-center gap-3">
                   <h3 className="text-xl font-black text-slate-800 thai-text leading-tight tracking-tight">{node.name_th}</h3>
                   <span className="px-3 py-1 bg-indigo-50 rounded-full text-xs font-bold text-indigo-400 uppercase tracking-wider border border-indigo-100/50">Level {level}</span>
                </div>
                {node.name_en && (
                  <p className="text-xs font-bold text-indigo-400 uppercase tracking-wider mt-1 italic">{node.name_en}</p>
                )}
                {node.description && (
                  <p className="text-sm text-slate-500 mt-3 thai-text leading-relaxed bg-white/40 border border-slate-100/50 p-4 rounded-2xl italic">
                    "{node.description}"
                  </p>
                )}
              </div>
            </div>
            
            <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-all transform translate-x-4 group-hover:translate-x-0 duration-300">
              <button
                onClick={() => handleCategoryEdit(node)}
                className="w-12 h-12 flex items-center justify-center bg-white border border-slate-100 text-slate-400 hover:text-indigo-600 hover:border-indigo-200 hover:bg-indigo-50 rounded-2xl shadow-sm transition-all"
              >
                <EditIcon className="h-5 w-5" />
              </button>
              <button
                onClick={() => handleCategoryDelete(node)}
                className="w-12 h-12 flex items-center justify-center bg-white border border-slate-100 text-slate-400 hover:text-rose-600 hover:border-rose-200 hover:bg-rose-50 rounded-2xl shadow-sm transition-all"
              >
                <TrashIcon className="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>
        
        {node.children && node.children.length > 0 && (
          <div className="mt-6">
            {renderCategoryTree(node.children, level + 1)}
          </div>
        )}
      </motion.div>
    ))
  }

  if (isLoading) {
    return (
      <div className="flex h-screen bg-white">
        <Sidebar />
        <div className="flex-1 flex flex-col items-center justify-center">
          <div className="w-20 h-20 relative flex items-center justify-center">
             <div className="absolute inset-0 border-4 border-indigo-100 rounded-full" />
             <div className="absolute inset-0 border-4 border-indigo-600 rounded-full border-t-transparent animate-spin" />
          </div>
          <p className="mt-8 text-xs font-black text-slate-400 uppercase tracking-[0.3em] thai-text">Scanning Taxonomy Tree...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-screen bg-gray-50 font-sans">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden relative">
        {/* Decorative elements */}
        <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-indigo-50/40 rounded-full blur-[120px] -mr-64 -mt-64 pointer-events-none" />
        <div className="absolute bottom-10 left-10 w-[400px] h-[400px] bg-emerald-50/40 rounded-full blur-[120px] pointer-events-none" />
        
        <Header />
        
        <main className="flex-1 overflow-x-hidden overflow-y-auto p-10 relative z-10">
          <div className="max-w-7xl mx-auto">
            {/* Page Header Area */}
            <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between mb-16 gap-8">
              <div>
                 <div className="flex items-center gap-2 mb-4">
                    <span className="w-2 h-2 rounded-full bg-indigo-500" />
                    <span className="text-xs font-bold text-slate-400 uppercase tracking-wider leading-relaxed">Hierarchical Intelligence</span>
                 </div>
                 <h1 className="text-3xl font-black text-slate-900 tracking-tight uppercase thai-text">
                   Taxonomy Master
                 </h1>
                 <p className="text-slate-500 mt-4 text-lg font-medium thai-text max-w-2xl leading-relaxed">
                   บริหารจัดการโครงสร้างและจัดระเบียบหมวดหมู่สินค้าแบบลำดับชั้น (Tree Structure) เพื่อเพิ่มประสิทธิภาพในการจับคู่และค้นหาข้อมูลด้วย AI
                 </p>
              </div>

              <div className="flex items-center gap-4">
                 <div className="px-8 py-3 bg-white border border-slate-100 rounded-3xl flex items-center gap-4 shadow-sm">
                    <div className="flex flex-col items-end">
                       <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Global Nodes</span>
                       <span className="text-xl font-black text-slate-900">{categories.length}</span>
                    </div>
                    <div className="w-[1px] h-8 bg-slate-100" />
                    <ActivityIcon className="w-5 h-5 text-emerald-500" />
                 </div>
                 <button
                    onClick={() => {
                      setEditingCategory(null)
                      setFormData({
                        name_th: '',
                        name_en: '',
                        description: '',
                        parent_id: '',
                        keywords: []
                      })
                      setShowForm(true)
                    }}
                    className="btn-premium px-10 py-5 h-auto text-base group"
                 >
                    <PlusIcon className="h-5 w-5 mr-3 group-hover:rotate-180 transition-transform duration-500" />
                    <span className="thai-text uppercase font-black tracking-widest">Create Node</span>
                 </button>
              </div>
            </div>

            {/* Content Container */}
            <div className="premium-card p-12 bg-white/40 border-white shadow-xl relative overflow-hidden">
              <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-50/10 rounded-full blur-[80px]" />
              
              {categories.length === 0 ? (
                <div className="text-center py-24 flex flex-col items-center">
                  <div className="w-24 h-24 rounded-full bg-slate-50 flex items-center justify-center mb-8 border border-slate-100">
                    <FolderTreeIcon className="h-10 w-10 text-slate-200" />
                  </div>
                  <h3 className="text-2xl font-black text-slate-900 mb-2 uppercase tracking-tight thai-text">Tree Empty</h3>
                  <p className="text-slate-400 font-bold uppercase tracking-widest text-[10px] mb-10">Initializing Root Node Required</p>
                  <button
                    onClick={() => setShowForm(true)}
                    className="btn-premium-secondary px-12 py-5 cursor-pointer font-black uppercase tracking-widest"
                  >
                    Generate First Node
                  </button>
                </div>
              ) : (
                <div className="space-y-4">
                  {renderCategoryTree(categories)}
                </div>
              )}
            </div>
          </div>
        </main>
      </div>

      {/* Form Modal Overhaul */}
      <AnimatePresence>
        {showForm && (
          <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-md flex items-center justify-center z-50 p-6">
            <motion.div
              initial={{ opacity: 0, y: 50, scale: 0.9, rotateX: 10 }}
              animate={{ opacity: 1, y: 0, scale: 1, rotateX: 0 }}
              exit={{ opacity: 0, y: 50, scale: 0.9, rotateX: -10 }}
              className="bg-white rounded-[56px] shadow-[0_40px_100px_-20px_rgba(0,0,0,0.3)] p-12 w-full max-w-2xl border border-white relative overflow-hidden"
            >
              <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-50 rounded-full blur-[80px] -mr-32 -mt-32 pointer-events-none opacity-50" />
              
              <div className="flex justify-between items-start mb-12 relative z-10">
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <span className="w-3 h-3 rounded-full bg-indigo-600 animate-pulse" />
                    <span className="text-[10px] font-black text-indigo-600 uppercase tracking-[.3em]">Neural Interface</span>
                  </div>
                  <h3 className="text-4xl font-black text-slate-900 tracking-tighter uppercase thai-text">
                    {editingCategory ? 'Edit Node' : 'Provison Node'}
                  </h3>
                </div>
                <button
                  onClick={() => setShowForm(false)}
                  className="w-14 h-14 flex items-center justify-center bg-slate-50 text-slate-400 hover:text-slate-900 hover:bg-slate-100 rounded-3xl transition-all"
                >
                  <XIcon className="h-6 w-6" />
                </button>
              </div>

              <form onSubmit={handleFormSubmit} className="space-y-10 relative z-10">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
                  <div className="space-y-3">
                    <label className="block text-[11px] font-black text-slate-400 uppercase tracking-[.2em] pl-2">
                      Local Identity (TH) *
                    </label>
                    <input
                      type="text"
                      value={formData.name_th}
                      onChange={(e) => setFormData({ ...formData, name_th: e.target.value })}
                      className="w-full px-8 py-5 bg-slate-50 border border-slate-100 rounded-[32px] focus:outline-none focus:ring-4 focus:ring-indigo-500/10 focus:border-indigo-500/50 transition-all thai-text font-black text-slate-800 placeholder:text-slate-300"
                      placeholder="เช่น อุปกรณ์เครื่องเขียน"
                      required
                    />
                  </div>

                  <div className="space-y-3">
                    <label className="block text-[11px] font-black text-slate-400 uppercase tracking-[.2em] pl-2">
                      Global Identity (EN)
                    </label>
                    <input
                      type="text"
                      value={formData.name_en}
                      onChange={(e) => setFormData({ ...formData, name_en: e.target.value })}
                      className="w-full px-8 py-5 bg-slate-50 border border-slate-100 rounded-[32px] focus:outline-none focus:ring-4 focus:ring-indigo-500/10 focus:border-indigo-500/50 transition-all font-black text-slate-800 placeholder:text-slate-300"
                      placeholder="e.g. Stationery"
                    />
                  </div>
                </div>

                <div className="space-y-3">
                  <label className="block text-[11px] font-black text-slate-400 uppercase tracking-[.2em] pl-2">
                    Scope & Description
                  </label>
                  <textarea
                    value={formData.description}
                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    className="w-full px-8 py-6 bg-slate-50 border border-slate-100 rounded-[40px] focus:outline-none focus:ring-4 focus:ring-indigo-500/10 focus:border-indigo-500/50 transition-all thai-text font-black text-slate-800 h-40 resize-none placeholder:text-slate-300"
                    placeholder="Describe the category scope..."
                  />
                </div>

                <div className="flex gap-6 pt-10">
                  <button
                    type="button"
                    onClick={() => setShowForm(false)}
                    className="flex-1 py-6 bg-white border border-slate-200 rounded-[32px] font-black text-slate-400 uppercase tracking-widest hover:bg-slate-50 hover:text-slate-600 transition-all"
                  >
                    Cancel Action
                  </button>
                  <button
                    type="submit"
                    className="flex-1 btn-premium py-6 rounded-[32px] font-black uppercase tracking-widest shadow-2xl shadow-indigo-200"
                  >
                    <SaveIcon className="h-5 w-5 mr-3" />
                    Commit Node
                  </button>
                </div>
              </form>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  )
}

