'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { toast } from 'react-hot-toast'
import Modal, { FOCUS_RING } from '@/components/ui/Modal'
import { 
  DatabaseService, 
  Synonym, 
  SynonymTerm,
  TaxonomyNode 
} from '@/utils/supabase'
import { 
  PlusIcon, 
  EditIcon, 
  TrashIcon, 
  SaveIcon, 
  XIcon,
  BookOpenIcon,
  SearchIcon,
  FilterIcon,
  CheckCircleIcon,
  XCircleIcon,
  TagIcon,
  SparklesIcon,
  LanguagesIcon,
  LayersIcon,
  ShieldCheckIcon
} from 'lucide-react'

interface SynonymFormData {
  code: string
  name_th: string
  description: string
  category_id?: string
  terms: Array<{
    term: string
    is_primary: boolean
    confidence_score: number
    source: 'manual' | 'auto' | 'imported' | 'ml'
    language: string
  }>
}

/**
 * งานเขียน synonym ต้องผ่าน API route ที่ใช้ service role — เขียนตรงด้วย anon key
 * จะถูก RLS ของ synonym_lemmas / synonym_terms ปฏิเสธ หรือเงียบแบบไม่แตะข้อมูล
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

export default function SynonymsPanel() {
  const [synonyms, setSynonyms] = useState<Synonym[]>([])
  const [categories, setCategories] = useState<TaxonomyNode[]>([])
  const [filteredSynonyms, setFilteredSynonyms] = useState<Synonym[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [showForm, setShowForm] = useState(false)
  const [editingSynonym, setEditingSynonym] = useState<Synonym | null>(null)
  
  // Filters
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<string>('')
  const [sourceFilter, setSourceFilter] = useState<string>('')
  
  // Form data
  const [formData, setFormData] = useState<SynonymFormData>({
    code: '',
    name_th: '',
    description: '',
    category_id: '',
    terms: []
  })

  useEffect(() => {
    loadData()
  }, [])

  useEffect(() => {
    filterSynonyms()
  }, [synonyms, searchTerm, selectedCategory, sourceFilter])

  const loadData = async () => {
    try {
      setIsLoading(true)
      const [synonymData, taxonomyData] = await Promise.all([
        DatabaseService.getSynonyms(),
        DatabaseService.getTaxonomyTree()
      ])
      setSynonyms(synonymData || [])
      setCategories(taxonomyData || [])
    } catch (error) {
      console.error('Error loading data:', error)
      toast.error('เกิดข้อผิดพลาดในการโหลดข้อมูล')
    } finally {
      setIsLoading(false)
    }
  }

  const filterSynonyms = () => {
    let filtered = synonyms
    if (searchTerm) {
      filtered = filtered.filter(synonym => 
        synonym.name_th.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (synonym.code && synonym.code.toLowerCase().includes(searchTerm.toLowerCase())) ||
        synonym.terms?.some(term => term.term.toLowerCase().includes(searchTerm.toLowerCase()))
      )
    }
    if (selectedCategory) filtered = filtered.filter(synonym => synonym.category_id === selectedCategory)
    if (sourceFilter) filtered = filtered.filter(synonym => synonym.terms?.some(term => term.source === sourceFilter))
    setFilteredSynonyms(filtered)
  }

  const handleSynonymAdd = () => {
    setEditingSynonym(null)
    setFormData({ code: '', name_th: '', description: '', category_id: '', terms: [] })
    setShowForm(true)
  }

  const handleSynonymEdit = (synonym: Synonym) => {
    setEditingSynonym(synonym)
    setFormData({
      code: synonym.code || '',
      name_th: synonym.name_th,
      description: synonym.description || '',
      category_id: synonym.category_id || '',
      terms: synonym.terms?.map(term => ({
        term: term.term,
        is_primary: Boolean(term.is_primary),
        confidence_score: Number(term.confidence_score ?? 1.0),
        source: (term.source as 'manual' | 'auto' | 'imported' | 'ml') || 'manual',
        language: term.language || 'th'
      })) || []
    })
    setShowForm(true)
  }

  const handleSynonymDelete = async (synonym: Synonym) => {
    if (!confirm('คุณแน่ใจหรือไม่ที่จะลบ synonym นี้?')) return
    try {
      await callApi(`/api/synonyms/${synonym.id}`, { method: 'DELETE' })
      toast.success('ลบ synonym เรียบร้อยแล้ว')
      loadData()
    } catch (error) {
      toast.error('เกิดข้อผิดพลาดในการลบ synonym')
    }
  }

  const handleFormSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      if (editingSynonym) {
        await callApi(`/api/synonyms/${editingSynonym.id}`, {
          method: 'PUT',
          body: JSON.stringify({
            code: formData.code || `SYN-${Date.now()}`,
            name_th: formData.name_th,
            description: formData.description,
            category_id: formData.category_id || undefined
          })
        })
        toast.success('อัปเดต synonym เรียบร้อยแล้ว')
      } else {
        // route เดียวจบ — สร้าง lemma พร้อม terms ในคำขอเดียว
        await callApi('/api/synonyms', {
          method: 'POST',
          body: JSON.stringify({
            code: formData.code || undefined,
            name_th: formData.name_th,
            description: formData.description,
            category_id: formData.category_id || undefined,
            terms: formData.terms
          })
        })
        toast.success('สร้าง synonym เรียบร้อยแล้ว')
      }

      setShowForm(false)
      loadData()
    } catch (error) {
      console.error('Save error:', error)
      toast.error('เกิดข้อผิดพลาดในการบันทึกข้อมูล')
    }
  }

  const addTerm = () => {
    setFormData(prev => ({
      ...prev,
      terms: [...prev.terms, {
        term: '',
        is_primary: prev.terms.length === 0,
        confidence_score: 1.0,
        source: 'manual',
        language: 'th'
      }]
    }))
  }

  if (isLoading) return <div data-testid="loading-indicator" className="flex items-center justify-center h-screen bg-gray-50"><div className="animate-spin rounded-full h-12 w-12 border-4 border-indigo-500 border-t-transparent shadow-xl"></div></div>

  return (
    <div className="h-full bg-gray-50 font-sans">
      <div className="flex-1 flex flex-col overflow-hidden">
        <main className="flex-1 overflow-x-hidden overflow-y-auto p-8 relative">
          {/* Decorative Background Elements */}
          <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-indigo-50/50 rounded-full blur-[120px] -mr-48 -mt-48 pointer-events-none" />
          <div className="absolute bottom-10 left-10 w-[300px] h-[300px] bg-emerald-50/50 rounded-full blur-[100px] pointer-events-none" />

          <div className="max-w-7xl mx-auto relative z-10">
            {/* Page Header */}
            <div className="flex flex-col md:flex-row md:items-end md:justify-between mb-12 gap-6">
              <div>
                <h1 data-testid="page-title" className="text-3xl font-black text-slate-900 tracking-tight thai-text uppercase">
                  Synonym Management Infrastructure
                </h1>
                <p className="mt-2 text-slate-500 font-medium thai-text">
                  บริหารจัดการคลังคำพ้องความหมายเพื่อเพิ่มความแม่นยำในการค้นหาและจัดหมวดหมู่ด้วย AI
                </p>
              </div>
              <button data-testid="add-synonym-btn" onClick={handleSynonymAdd} className={`btn-premium px-10 h-14 group ${FOCUS_RING}`}>
                <PlusIcon className="w-5 h-5 mr-2 group-hover:rotate-90 transition-transform duration-300" aria-hidden="true" />
                <span className="thai-text">Add New Synonym</span>
              </button>
            </div>

            {/* Quick Stats */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 mb-10">
               <div className="premium-card p-6 flex items-center gap-6 bg-white/60">
                  <div className="w-14 h-14 bg-indigo-50 rounded-2xl flex items-center justify-center text-indigo-600 shadow-sm border border-indigo-100/50">
                     <BookOpenIcon className="w-6 h-6" aria-hidden="true" />
                  </div>
                  <div>
                    <h3 className="text-3xl font-black text-slate-900 leading-none">{synonyms.length}</h3>
                    <p className="text-xs font-bold text-slate-600 uppercase tracking-wider mt-1">Total Sets</p>
                  </div>
               </div>
               <div className="premium-card p-6 flex items-center gap-6 bg-white/60">
                  <div className="w-14 h-14 bg-emerald-50 rounded-2xl flex items-center justify-center text-emerald-600 shadow-sm border border-emerald-100/50">
                     <SparklesIcon className="w-6 h-6" aria-hidden="true" />
                  </div>
                  <div>
                    <h3 className="text-3xl font-black text-slate-900 leading-none">
                      {synonyms.reduce((acc, s) => acc + (s.terms?.length || 0), 0)}
                    </h3>
                    <p className="text-xs font-bold text-slate-600 uppercase tracking-wider mt-1">Bound Terms</p>
                  </div>
               </div>
               <div className="premium-card p-6 flex items-center gap-6 bg-white/60">
                  <div className="w-14 h-14 bg-amber-50 rounded-2xl flex items-center justify-center text-amber-700 shadow-sm border border-amber-100/50">
                     <ShieldCheckIcon className="w-6 h-6" aria-hidden="true" />
                  </div>
                  <div>
                    <h3 className="text-3xl font-black text-slate-900 leading-none">
                      {synonyms.filter(s => s.is_active).length}
                    </h3>
                    <p className="text-xs font-bold text-slate-600 uppercase tracking-wider mt-1">Active Rules</p>
                  </div>
               </div>
            </div>

            {/* Controls */}
            <div className="p-2.5 bg-white/40 backdrop-blur-md rounded-3xl border border-white/80 shadow-sm mb-10">
              <div className="flex flex-col lg:flex-row gap-4 items-center">
                <div className="relative flex-1 w-full lg:w-auto">
                  <label htmlFor="synonym-search-input" className="sr-only">ค้นหาคำพ้องความหมาย</label>
                  <SearchIcon className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500 w-5 h-5 pointer-events-none" aria-hidden="true" />
                  <input
                    id="synonym-search-input"
                    type="text"
                    data-testid="search-input"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    placeholder="Search synonym library..."
                    className={`w-full pl-12 pr-4 py-3 bg-white/70 border border-slate-200 rounded-2xl focus:outline-none focus:bg-white transition-all text-sm font-bold text-slate-800 placeholder:text-slate-500 uppercase tracking-wider ${FOCUS_RING}`}
                  />
                </div>
                
                <div className="flex gap-4 w-full lg:w-auto overflow-x-auto pb-1 lg:pb-0">
                  <label htmlFor="synonym-category-filter" className="sr-only">กรองตามหมวดหมู่</label>
                  <select
                    id="synonym-category-filter"
                    value={selectedCategory}
                    data-testid="category-filter"
                    onChange={(e) => setSelectedCategory(e.target.value)}
                    className={`select-premium text-xs font-bold tracking-wider uppercase min-w-[180px] ${FOCUS_RING}`}
                  >
                    <option value="">All Categories</option>
                    {categories.map(c => <option key={c.id} value={c.id}>{c.name_th}</option>)}
                  </select>

                  <label htmlFor="synonym-source-filter" className="sr-only">กรองตามแหล่งที่มา</label>
                  <select
                    id="synonym-source-filter"
                    value={sourceFilter}
                    onChange={(e) => setSourceFilter(e.target.value)}
                    className={`select-premium text-xs font-bold tracking-wider uppercase ${FOCUS_RING}`}
                  >
                    <option value="">Any Source</option>
                    <option value="manual">Manual</option>
                    <option value="ml">AI Suggested</option>
                    <option value="imported">External</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Content Table */}
            <div className="premium-card overflow-hidden bg-white/60 backdrop-blur-xl border-white translate-z-0">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <caption className="sr-only">ตารางกลุ่มคำพ้องความหมาย (Synonyms)</caption>
                  <thead>
                    <tr className="border-b border-slate-200 bg-slate-50/80">
                      <th scope="col" className="px-10 py-6 text-left text-xs font-bold text-slate-700 uppercase tracking-wider">Synonym Group</th>
                      <th scope="col" className="px-10 py-6 text-left text-xs font-bold text-slate-700 uppercase tracking-wider">Connected Nodes</th>
                      <th scope="col" className="px-10 py-6 text-left text-xs font-bold text-slate-700 uppercase tracking-wider">Taxonomy Alignment</th>
                      <th scope="col" className="px-10 py-6 text-left text-xs font-bold text-slate-700 uppercase tracking-wider">Status</th>
                      <th scope="col" className="px-10 py-6 text-right text-xs font-bold text-slate-700 uppercase tracking-wider">Management</th>
                    </tr>
                  </thead>
                  <tbody data-testid="synonym-list" className="divide-y divide-slate-100">
                    <AnimatePresence mode="popLayout">
                      {filteredSynonyms.map((synonym, idx) => (
                        <motion.tr 
                          key={synonym.id}
                          data-testid={`synonym-item-${synonym.id}`}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, scale: 0.95 }}
                          transition={{ delay: idx * 0.03 }}
                          className="group hover:bg-indigo-50/20 transition-colors"
                        >
                          <td className="px-10 py-7">
                            <div className="flex items-center gap-5">
                              <div className="w-12 h-12 bg-indigo-50 rounded-2xl flex items-center justify-center text-indigo-600 border border-indigo-100/50 group-hover:scale-110 group-hover:rotate-3 transition-all duration-300">
                                <LayersIcon className="w-6 h-6" aria-hidden="true" />
                              </div>
                              <div>
                                <div data-testid="synonym-lemma" className="text-base font-black text-slate-800 thai-text tracking-tight">{synonym.name_th}</div>
                                <div className="text-xs font-bold text-slate-500 uppercase tracking-wider mt-1">CODE: {synonym.code}</div>
                              </div>
                            </div>
                          </td>
                          <td className="px-10 py-7">
                            <div className="flex flex-wrap gap-2">
                              {synonym.terms?.slice(0, 4).map((term, i) => (
                                <span key={i} className={`px-3 py-1 rounded-xl text-xs font-bold uppercase tracking-wider border transition-all ${
                                  term.is_primary ? 'bg-indigo-600 text-white border-indigo-600 shadow-md shadow-indigo-200' : 'bg-slate-100 text-slate-700 border-slate-300 group-hover:border-indigo-200'
                                }`}>
                                  {term.term}
                                </span>
                              ))}
                              {synonym.terms && synonym.terms.length > 4 && (
                                <span className="text-xs font-bold text-slate-500 p-1">+ {synonym.terms.length - 4} Others</span>
                              )}
                            </div>
                          </td>
                          <td className="px-10 py-7">
                            <div className="flex items-center gap-2">
                              {synonym.category ? (
                                <>
                                  <div className="w-2 h-2 bg-indigo-500 rounded-full motion-safe:animate-pulse" aria-hidden="true" />
                                  <span data-testid="synonym-category" className="text-xs font-black text-slate-700 thai-text">{synonym.category.name_th}</span>
                                </>
                              ) : (
                                <span className="text-xs font-medium text-slate-500 italic">Unmapped</span>
                              )}
                            </div>
                          </td>
                          <td className="px-10 py-7">
                             <div className={`inline-flex items-center px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider border ${
                               synonym.is_active ? 'bg-emerald-100 text-emerald-900 border-emerald-300' : 'bg-slate-100 text-slate-700 border-slate-300'
                             }`}>
                                {synonym.is_active ? 'Active Sync' : 'Offline'}
                             </div>
                          </td>
                          <td className="px-10 py-7 text-right">
                            <div className="flex items-center justify-end gap-3 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 focus-visible:opacity-100 transition-all">
                              <button 
                                type="button"
                                data-testid={`edit-synonym-${synonym.id}`} 
                                aria-label={`แก้ไขคำพ้อง ${synonym.name_th}`}
                                onClick={() => handleSynonymEdit(synonym)} 
                                className={`p-3 bg-white rounded-xl text-slate-600 hover:text-indigo-600 border border-slate-200 hover:border-indigo-200 hover:shadow-xl hover:shadow-indigo-100/20 transition-all ${FOCUS_RING}`}
                              >
                                <EditIcon className="w-4 h-4" aria-hidden="true" />
                              </button>
                              <button 
                                type="button"
                                data-testid={`delete-synonym-${synonym.id}`} 
                                aria-label={`ลบคำพ้อง ${synonym.name_th}`}
                                onClick={() => handleSynonymDelete(synonym)} 
                                className={`p-3 bg-white rounded-xl text-slate-600 hover:text-rose-600 border border-slate-200 hover:border-rose-200 hover:shadow-xl hover:shadow-rose-100/20 transition-all ${FOCUS_RING}`}
                              >
                                <TrashIcon className="w-4 h-4" aria-hidden="true" />
                              </button>
                            </div>
                          </td>
                        </motion.tr>
                      ))}
                    </AnimatePresence>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </main>
      </div>

      {/* Form Modal */}
      <Modal
        isOpen={showForm}
        onClose={() => setShowForm(false)}
        title={editingSynonym ? 'Configure Synonym' : 'Provision New Group'}
        maxWidth="max-w-4xl"
      >
        <form onSubmit={handleFormSubmit} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label htmlFor="synonym-code" className="block text-xs font-bold text-slate-700 uppercase tracking-wider">
                Infrastructure Code (Unique)
              </label>
              <input
                id="synonym-code"
                type="text"
                data-testid="code-input"
                value={formData.code}
                onChange={(e) => setFormData(prev => ({ ...prev, code: e.target.value }))}
                className={`input-premium w-full text-base font-bold uppercase tracking-wider ${FOCUS_RING}`}
                placeholder="e.g. SMARTPHONE_001"
              />
            </div>
            <div className="space-y-2">
              <label htmlFor="synonym-name-th" className="block text-xs font-bold text-slate-700 uppercase tracking-wider">
                Identifier Name (TH) <span className="text-rose-600">*</span>
              </label>
              <input
                id="synonym-name-th"
                type="text"
                required
                data-testid="lemma-input"
                value={formData.name_th}
                onChange={(e) => setFormData(prev => ({ ...prev, name_th: e.target.value }))}
                className={`input-premium w-full text-base font-bold thai-text ${FOCUS_RING}`}
                placeholder="e.g. สมาร์ทโฟน"
              />
            </div>
          </div>

          <div className="space-y-2">
            <label htmlFor="synonym-category-select" className="block text-xs font-bold text-slate-700 uppercase tracking-wider">
              Taxonomy Mapping
            </label>
            <select
              id="synonym-category-select"
              value={formData.category_id}
              data-testid="category-select"
              onChange={(e) => setFormData(prev => ({ ...prev, category_id: e.target.value }))}
              className={`select-premium w-full text-base font-bold thai-text ${FOCUS_RING}`}
            >
              <option value="">Select Anchor Category</option>
              {categories.map(c => <option key={c.id} value={c.id}>{c.name_th}</option>)}
            </select>
          </div>

          <div className="space-y-2">
            <label htmlFor="synonym-description" className="block text-xs font-bold text-slate-700 uppercase tracking-wider">
              Administrative Note
            </label>
            <textarea
              id="synonym-description"
              value={formData.description}
              onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
              className={`input-premium w-full h-28 text-sm font-medium thai-text resize-none ${FOCUS_RING}`}
              placeholder="Describe the logic or context for this synonym group..."
            />
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between border-b border-slate-200 pb-3">
              <label className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                Expansion Terms (Variations)
              </label>
              <button 
                type="button" 
                data-testid="add-term-btn" 
                onClick={addTerm} 
                className={`btn-premium-secondary py-2 px-4 text-xs font-bold ${FOCUS_RING}`}
              >
                <PlusIcon className="w-4 h-4 mr-1.5" aria-hidden="true" />
                Add Variant
              </button>
            </div>
            
            <div className="space-y-3 max-h-60 overflow-y-auto pr-1">
              {formData.terms.map((term, index) => (
                <div key={index} className="flex gap-3 items-center p-4 bg-slate-50 rounded-2xl border border-slate-200 group/term hover:bg-white hover:border-indigo-200 transition-all">
                  <div className="p-2.5 bg-white rounded-xl text-slate-500 border border-slate-200 group-hover/term:text-indigo-600 transition-colors">
                    <LanguagesIcon className="w-4 h-4" aria-hidden="true" />
                  </div>
                  <label htmlFor={`term-input-${index}`} className="sr-only">รูปแบบคำที่ {index + 1}</label>
                  <input
                    id={`term-input-${index}`}
                    type="text"
                    data-testid={`term-input-${index}`}
                    value={term.term}
                    onChange={(e) => {
                      const newTerms = [...formData.terms];
                      newTerms[index].term = e.target.value;
                      setFormData(p => ({ ...p, terms: newTerms }));
                    }}
                    placeholder="Term Variation"
                    className={`flex-1 bg-white border border-slate-200 rounded-xl px-4 py-2.5 text-sm font-bold text-slate-800 thai-text ${FOCUS_RING}`}
                  />
                  <div className="flex items-center gap-4 pr-1">
                    <label className="flex items-center gap-2 cursor-pointer group/check">
                      <input
                        type="checkbox"
                        id={`term-primary-${index}`}
                        aria-label={`กำหนดให้คำรูปแบบที่ ${index + 1} เป็นคำหลัก (Primary)`}
                        checked={term.is_primary}
                        onChange={(e) => {
                          const newTerms = formData.terms.map((t, i) => ({ 
                            ...t, 
                            is_primary: i === index ? e.target.checked : (e.target.checked ? false : t.is_primary) 
                          }));
                          setFormData(p => ({ ...p, terms: newTerms }));
                        }}
                        className={`w-4 h-4 text-indigo-600 rounded border-slate-300 ${FOCUS_RING}`}
                      />
                      <span className="text-xs font-bold text-slate-600 uppercase tracking-wider group-hover/check:text-indigo-600 transition-colors">Primary</span>
                    </label>
                    <button 
                      type="button"
                      aria-label={`ลบรูปแบบคำที่ ${index + 1}`}
                      onClick={() => setFormData(p => ({ ...p, terms: p.terms.filter((_, i) => i !== index) }))} 
                      className={`p-2 text-slate-400 hover:text-rose-600 rounded-lg transition-colors ${FOCUS_RING}`}
                    >
                      <XIcon className="w-4 h-4" aria-hidden="true" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="pt-6 border-t border-slate-200 flex justify-end gap-4">
            <button 
              type="button" 
              onClick={() => setShowForm(false)} 
              className={`btn-premium-secondary py-3 px-6 ${FOCUS_RING}`}
            >
              Discard Config
            </button>
            <button 
              type="submit" 
              data-testid="save-synonym-btn" 
              className={`btn-premium py-3 px-8 justify-center group ${FOCUS_RING}`}
            >
              <SaveIcon className="w-5 h-5 mr-2 group-hover:scale-110 transition-transform" aria-hidden="true" />
              <span className="thai-text text-base">Commit Infrastructure</span>
            </button>
          </div>
        </form>
      </Modal>
    </div>
  )
}

