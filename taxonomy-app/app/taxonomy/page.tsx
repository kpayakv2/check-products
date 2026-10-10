'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { toast } from 'react-hot-toast'
import Sidebar from '@/components/Layout/Sidebar'
import Header from '@/components/Layout/Header'
import SynonymsPanel from '@/components/Taxonomy/SynonymsPanel'
import Modal, { FOCUS_RING } from '@/components/ui/Modal'
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

/**
 * ทุกการเขียนต้องผ่าน API route ที่ใช้ service role — เขียนตรงด้วย anon key
 * จะถูก RLS ปฏิเสธ (เพิ่มไม่ได้) หรือ "สำเร็จ" แบบไม่แตะข้อมูล (แก้/ลบ)
 */
const callApi = async (url: string, init: RequestInit) => {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...init
  })
  const body = await response.json().catch(() => ({}))

  if (!response.ok || body?.success === false) {
    throw new Error(body?.error || `คำขอล้มเหลว (${response.status})`)
  }
  return body
}

/** นับทั้งต้นไม้ ไม่ใช่เฉพาะโหนดบนสุดที่ `categories` เก็บไว้ */
const countNodes = (nodes: TaxonomyNode[]): number =>
  nodes.reduce((total, node) => total + 1 + countNodes(node.children || []), 0)

export default function TaxonomyManager() {
  const [activeTab, setActiveTab] = useState<'tree' | 'synonyms'>('tree')
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
      await callApi(`/api/taxonomy/${category.id}`, { method: 'DELETE' })
      toast.success('ลบหมวดหมู่เรียบร้อยแล้ว')
      loadCategories()
    } catch (error) {
      console.error('Error deleting category:', error)
      toast.error(error instanceof Error ? error.message : 'เกิดข้อผิดพลาดในการลบหมวดหมู่')
    }
  }

  const handleFormSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!formData.name_th.trim()) {
      toast.error('กรุณาใส่ชื่อหมวดหมู่ภาษาไทย')
      return
    }

    try {
      // ช่องเลือกหมวดแม่ที่ไม่ได้เลือกจะเป็นสตริงว่าง ซึ่งคอลัมน์ uuid รับไม่ได้
      const payload = {
        ...formData,
        parent_id: formData.parent_id || undefined
      }

      if (editingCategory) {
        await callApi(`/api/taxonomy/${editingCategory.id}`, {
          method: 'PUT',
          body: JSON.stringify(payload)
        })
        toast.success('แก้ไขหมวดหมู่เรียบร้อยแล้ว')
      } else {
        await callApi('/api/taxonomy', {
          method: 'POST',
          body: JSON.stringify(payload)
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
      toast.error(error instanceof Error ? error.message : 'เกิดข้อผิดพลาดในการบันทึกหมวดหมู่')
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
            
            <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 focus-within:opacity-100 transition-all transform translate-x-4 group-hover:translate-x-0 group-focus-within:translate-x-0 duration-300">
              <button
                type="button"
                aria-label={`แก้ไขหมวดหมู่ ${node.name_th}`}
                onClick={() => handleCategoryEdit(node)}
                className={`w-12 h-12 flex items-center justify-center bg-white border border-slate-200 text-slate-500 hover:text-indigo-600 hover:border-indigo-300 hover:bg-indigo-50 rounded-2xl shadow-sm transition-all ${FOCUS_RING}`}
              >
                <EditIcon className="h-5 w-5" aria-hidden="true" />
              </button>
              <button
                type="button"
                data-testid="delete-node"
                aria-label={`ลบหมวดหมู่ ${node.name_th}`}
                onClick={() => handleCategoryDelete(node)}
                className={`w-12 h-12 flex items-center justify-center bg-white border border-slate-200 text-slate-500 hover:text-rose-600 hover:border-rose-300 hover:bg-rose-50 rounded-2xl shadow-sm transition-all ${FOCUS_RING}`}
              >
                <TrashIcon className="h-5 w-5" aria-hidden="true" />
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
        
        {/* Taxonomy Tabs Header */}
        <div className="bg-white border-b border-slate-200 px-8 py-4 z-20 relative shadow-sm">
          <div 
            role="tablist"
            aria-label="การจัดการโครงสร้างหมวดหมู่"
            className="max-w-7xl mx-auto flex gap-4 p-1.5 bg-slate-100/90 rounded-[20px] w-fit"
          >
            <button 
              id="taxonomy-tab-tree"
              role="tab"
              type="button"
              data-testid="tab-tree"
              aria-selected={activeTab === 'tree'}
              aria-controls="taxonomy-panel-tree"
              tabIndex={activeTab === 'tree' ? 0 : -1}
              onClick={() => setActiveTab('tree')}
              onKeyDown={(e) => {
                if (e.key === 'ArrowRight') {
                  e.preventDefault()
                  setActiveTab('synonyms')
                  document.getElementById('taxonomy-tab-synonyms')?.focus()
                }
              }}
              className={`px-8 py-3 rounded-[16px] text-sm font-bold transition-all flex items-center gap-2 thai-text uppercase ${FOCUS_RING} ${
                activeTab === 'tree' ? 'bg-white text-indigo-700 shadow-md border border-slate-200/80' : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
              }`}
            >
              <FolderTreeIcon className="w-4 h-4" aria-hidden="true" />
              โครงสร้างหมวดหมู่ (Taxonomy Tree)
            </button>
            <button 
              id="taxonomy-tab-synonyms"
              role="tab"
              type="button"
              data-testid="tab-synonyms"
              aria-selected={activeTab === 'synonyms'}
              aria-controls="taxonomy-panel-synonyms"
              tabIndex={activeTab === 'synonyms' ? 0 : -1}
              onClick={() => setActiveTab('synonyms')}
              onKeyDown={(e) => {
                if (e.key === 'ArrowLeft') {
                  e.preventDefault()
                  setActiveTab('tree')
                  document.getElementById('taxonomy-tab-tree')?.focus()
                }
              }}
              className={`px-8 py-3 rounded-[16px] text-sm font-bold transition-all flex items-center gap-2 thai-text uppercase ${FOCUS_RING} ${
                activeTab === 'synonyms' ? 'bg-white text-indigo-700 shadow-md border border-slate-200/80' : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
              }`}
            >
              <LayersIcon className="w-4 h-4" aria-hidden="true" />
              คำพ้องความหมาย (Synonyms)
            </button>
          </div>
        </div>

        {activeTab === 'synonyms' ? (
          <div 
            role="tabpanel"
            id="taxonomy-panel-synonyms"
            aria-labelledby="taxonomy-tab-synonyms"
            tabIndex={0}
            className="flex-1 overflow-hidden relative bg-gray-50/50 focus:outline-none"
          >
            <SynonymsPanel />
          </div>
        ) : (
          <main 
            role="tabpanel"
            id="taxonomy-panel-tree"
            aria-labelledby="taxonomy-tab-tree"
            tabIndex={0}
            className="flex-1 overflow-x-hidden overflow-y-auto p-10 relative z-10 focus:outline-none"
          >
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
                       <span data-testid="node-count" className="text-xl font-black text-slate-900">{countNodes(categories)}</span>
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
                  <p className="text-slate-400 font-bold uppercase tracking-widest text-xs mb-10">Initializing Root Node Required</p>
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
        )}
      </div>

      {/* Category Form Modal */}
      <Modal
        isOpen={showForm}
        onClose={() => setShowForm(false)}
        title={editingCategory ? 'แก้ไขหมวดหมู่ (Edit Node)' : 'สร้างหมวดหมู่ใหม่ (Create Node)'}
        titleId="taxonomy-modal-title"
        maxWidth="max-w-2xl"
        className="rounded-[36px]"
      >
        <form onSubmit={handleFormSubmit} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label htmlFor="taxonomy-name-th" className="block text-xs font-bold text-slate-700 uppercase tracking-wider">
                ชื่อหมวดหมู่ภาษาไทย (TH) *
              </label>
              <input
                id="taxonomy-name-th"
                type="text"
                value={formData.name_th}
                onChange={(e) => setFormData({ ...formData, name_th: e.target.value })}
                className={`w-full px-5 py-3.5 bg-slate-50 border border-slate-200 rounded-2xl thai-text font-bold text-slate-800 placeholder:text-slate-400 text-sm ${FOCUS_RING}`}
                placeholder="เช่น อุปกรณ์เครื่องเขียน"
                required
              />
            </div>

            <div className="space-y-2">
              <label htmlFor="taxonomy-name-en" className="block text-xs font-bold text-slate-700 uppercase tracking-wider">
                ชื่อภาษาอังกฤษ (EN)
              </label>
              <input
                id="taxonomy-name-en"
                type="text"
                value={formData.name_en}
                onChange={(e) => setFormData({ ...formData, name_en: e.target.value })}
                className={`w-full px-5 py-3.5 bg-slate-50 border border-slate-200 rounded-2xl font-bold text-slate-800 placeholder:text-slate-400 text-sm ${FOCUS_RING}`}
                placeholder="e.g. Stationery"
              />
            </div>
          </div>

          <div className="space-y-2">
            <label htmlFor="taxonomy-desc" className="block text-xs font-bold text-slate-700 uppercase tracking-wider">
              คำอธิบายและขอบเขตหมวดหมู่ (Description)
            </label>
            <textarea
              id="taxonomy-desc"
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              className={`w-full px-5 py-4 bg-slate-50 border border-slate-200 rounded-2xl thai-text font-medium text-slate-800 h-32 resize-none placeholder:text-slate-400 text-sm ${FOCUS_RING}`}
              placeholder="ระบุรายละเอียดหรือขอบเขตของหมวดหมู่นี้..."
            />
          </div>

          <div className="flex gap-4 pt-4 border-t border-slate-100">
            <button
              type="button"
              onClick={() => setShowForm(false)}
              className={`flex-1 py-3.5 bg-white border border-slate-200 rounded-2xl font-bold text-slate-700 uppercase tracking-wider hover:bg-slate-50 transition-all text-xs ${FOCUS_RING}`}
            >
              ยกเลิก (Cancel)
            </button>
            <button
              type="submit"
              className={`flex-1 btn-premium py-3.5 rounded-2xl font-bold uppercase tracking-wider shadow-lg shadow-indigo-100 text-xs ${FOCUS_RING}`}
            >
              <SaveIcon className="h-4 w-4 mr-2" aria-hidden="true" />
              บันทึกหมวดหมู่ (Commit Node)
            </button>
          </div>
        </form>
      </Modal>
    </div>
  )
}

