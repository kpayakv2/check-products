'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { toast } from 'react-hot-toast'
import Sidebar from '@/components/Layout/Sidebar'
import Header from '@/components/Layout/Header'
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
  name: string
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

export default function SynonymsPage() {
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
    name: '',
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
        synonym.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        synonym.terms?.some(term => term.term.toLowerCase().includes(searchTerm.toLowerCase()))
      )
    }
    if (selectedCategory) filtered = filtered.filter(synonym => synonym.category_id === selectedCategory)
    if (sourceFilter) filtered = filtered.filter(synonym => synonym.terms?.some(term => term.source === sourceFilter))
    setFilteredSynonyms(filtered)
  }

  const handleSynonymAdd = () => {
    setEditingSynonym(null)
    setFormData({ name: '', description: '', category_id: '', terms: [] })
    setShowForm(true)
  }

  const handleSynonymEdit = (synonym: Synonym) => {
    setEditingSynonym(synonym)
    setFormData({
      name: synonym.name,
      description: synonym.description || '',
      category_id: synonym.category_id || '',
      terms: synonym.terms?.map(term => ({
        term: term.term,
        is_primary: term.is_primary,
        confidence_score: term.confidence_score,
        source: term.source,
        language: term.language
      })) || []
    })
    setShowForm(true)
  }

  const handleSynonymDelete = async (synonym: Synonym) => {
    if (!confirm('คุณแน่ใจหรือไม่ที่จะลบ synonym นี้?')) return
    try {
      await DatabaseService.deleteSynonym(synonym.id)
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
        await DatabaseService.updateSynonym(editingSynonym.id, {
          name: formData.name,
          description: formData.description,
          category_id: formData.category_id
        })
        toast.success('อัปเดต synonym เรียบร้อยแล้ว')
      } else {
        const newSynonym = await DatabaseService.createSynonym({
          name: formData.name,
          description: formData.description,
          category_id: formData.category_id,
          is_active: true
        })
        for (const termData of formData.terms) {
          await DatabaseService.createSynonymTerm({
            synonym_id: newSynonym.id,
            term: termData.term,
            is_primary: termData.is_primary,
            confidence_score: termData.confidence_score,
            source: termData.source,
            language: termData.language,
            is_verified: termData.source === 'manual'
          })
        }
        toast.success('สร้าง synonym เรียบร้อยแล้ว')
      }
      setShowForm(false)
      loadData()
    } catch (error) {
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

  if (isLoading) return <div className="flex items-center justify-center h-screen bg-gray-50"><div className="animate-spin rounded-full h-12 w-12 border-4 border-indigo-500 border-t-transparent shadow-xl"></div></div>

  return (
    <div className="flex h-screen bg-gray-50 font-sans">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        
        <main className="flex-1 overflow-x-hidden overflow-y-auto p-8 relative">
          {/* Decorative Background Elements */}
          <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-indigo-50/50 rounded-full blur-[120px] -mr-48 -mt-48 pointer-events-none" />
          <div className="absolute bottom-10 left-10 w-[300px] h-[300px] bg-emerald-50/50 rounded-full blur-[100px] pointer-events-none" />

          <div className="max-w-7xl mx-auto relative z-10">
            {/* Page Header */}
            <div className="flex flex-col md:flex-row md:items-end md:justify-between mb-12 gap-6">
              <div>
                <h1 className="text-3xl font-black text-slate-900 tracking-tight thai-text uppercase">
                  Synonym Infrastructure
                </h1>
                <p className="mt-2 text-slate-500 font-medium thai-text">
                  บริหารจัดการคลังคำพ้องความหมายเพื่อเพิ่มความแม่นยำในการค้นหาและจัดหมวดหมู่ด้วย AI
                </p>
              </div>
              <button onClick={handleSynonymAdd} className="btn-premium px-10 h-14 group">
                <PlusIcon className="w-5 h-5 mr-2 group-hover:rotate-90 transition-transform duration-300" />
                <span className="thai-text">Add New Synonym</span>
              </button>
            </div>

            {/* Quick Stats */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 mb-10">
               <div className="premium-card p-6 flex items-center gap-6 bg-white/60">
                  <div className="w-14 h-14 bg-indigo-50 rounded-2xl flex items-center justify-center text-indigo-600 shadow-sm border border-indigo-100/50">
                     <BookOpenIcon className="w-6 h-6" />
                  </div>
                  <div>
                    <h3 className="text-3xl font-black text-slate-900 leading-none">{synonyms.length}</h3>
                    <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mt-1">Total Sets</p>
                  </div>
               </div>
               <div className="premium-card p-6 flex items-center gap-6 bg-white/60">
                  <div className="w-14 h-14 bg-emerald-50 rounded-2xl flex items-center justify-center text-emerald-600 shadow-sm border border-emerald-100/50">
                     <SparklesIcon className="w-6 h-6" />
                  </div>
                  <div>
                    <h3 className="text-3xl font-black text-slate-900 leading-none">
                      {synonyms.reduce((acc, s) => acc + (s.terms?.length || 0), 0)}
                    </h3>
                    <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mt-1">Bound Terms</p>
                  </div>
               </div>
               <div className="premium-card p-6 flex items-center gap-6 bg-white/60">
                  <div className="w-14 h-14 bg-amber-50 rounded-2xl flex items-center justify-center text-amber-600 shadow-sm border border-amber-100/50">
                     <ShieldCheckIcon className="w-6 h-6" />
                  </div>
                  <div>
                    <h3 className="text-3xl font-black text-slate-900 leading-none">
                      {synonyms.filter(s => s.is_active).length}
                    </h3>
                    <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mt-1">Active Rules</p>
                  </div>
               </div>
            </div>

            {/* Controls */}
            <div className="p-2.5 bg-white/40 backdrop-blur-md rounded-3xl border border-white/80 shadow-sm mb-10">
              <div className="flex flex-col lg:flex-row gap-4 items-center">
                <div className="relative flex-1 w-full lg:w-auto">
                  <SearchIcon className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 w-5 h-5" />
                  <input
                    type="text"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    placeholder="Search synonym library..."
                    className="w-full pl-12 pr-4 py-3 bg-white/50 border border-slate-100 rounded-2xl focus:outline-none focus:ring-4 focus:ring-indigo-500/10 focus:bg-white transition-all text-sm font-black text-slate-700 uppercase tracking-widest"
                  />
                </div>
                
                <div className="flex gap-4 w-full lg:w-auto overflow-x-auto pb-1 lg:pb-0">
                  <select
                    value={selectedCategory}
                    onChange={(e) => setSelectedCategory(e.target.value)}
                    className="select-premium text-xs font-bold tracking-wider uppercase min-w-[180px]"
                  >
                    <option value="">All Categories</option>
                    {categories.map(c => <option key={c.id} value={c.id}>{c.name_th}</option>)}
                  </select>

                  <select
                    value={sourceFilter}
                    onChange={(e) => setSourceFilter(e.target.value)}
                    className="select-premium text-[10px] font-black tracking-widest uppercase"
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
                  <thead>
                    <tr className="border-b border-slate-100 bg-slate-50/30">
                      <th className="px-10 py-6 text-left text-[10px] font-black text-slate-400 uppercase tracking-[0.2em]">Synonym Group</th>
                      <th className="px-10 py-6 text-left text-[10px] font-black text-slate-400 uppercase tracking-[0.2em]">Connected Nodes</th>
                      <th className="px-10 py-6 text-left text-[10px] font-black text-slate-400 uppercase tracking-[0.2em]">Taxonomy Alignment</th>
                      <th className="px-10 py-6 text-left text-[10px] font-black text-slate-400 uppercase tracking-[0.2em]">Status</th>
                      <th className="px-10 py-6 text-right text-[10px] font-black text-slate-400 uppercase tracking-[0.2em]">Management</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    <AnimatePresence mode="popLayout">
                      {filteredSynonyms.map((synonym, idx) => (
                        <motion.tr 
                          key={synonym.id}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, scale: 0.95 }}
                          transition={{ delay: idx * 0.03 }}
                          className="group hover:bg-indigo-50/20 transition-colors"
                        >
                          <td className="px-10 py-7">
                            <div className="flex items-center gap-5">
                              <div className="w-12 h-12 bg-indigo-50 rounded-2xl flex items-center justify-center text-indigo-600 border border-indigo-100/50 group-hover:scale-110 group-hover:rotate-3 transition-all duration-300">
                                <LayersIcon className="w-6 h-6" />
                              </div>
                              <div>
                                <div className="text-base font-black text-slate-800 thai-text tracking-tight">{synonym.name}</div>
                                <div className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mt-1">ID: {synonym.id.slice(0, 8)}</div>
                              </div>
                            </div>
                          </td>
                          <td className="px-10 py-7">
                            <div className="flex flex-wrap gap-2">
                              {synonym.terms?.slice(0, 4).map((term, i) => (
                                <span key={i} className={`px-3 py-1 rounded-xl text-[10px] font-black uppercase tracking-widest border transition-all ${
                                  term.is_primary ? 'bg-indigo-600 text-white border-indigo-600 shadow-md shadow-indigo-200' : 'bg-white text-slate-500 border-slate-100 group-hover:border-indigo-100'
                                }`}>
                                  {term.term}
                                </span>
                              ))}
                              {synonym.terms && synonym.terms.length > 4 && (
                                <span className="text-[10px] font-black text-slate-300 p-1">+ {synonym.terms.length - 4} Others</span>
                              )}
                            </div>
                          </td>
                          <td className="px-10 py-7">
                            <div className="flex items-center gap-2">
                              {synonym.category ? (
                                <>
                                  <div className="w-2 h-2 bg-indigo-400 rounded-full animate-pulse" />
                                  <span className="text-xs font-black text-slate-600 thai-text">{synonym.category.name_th}</span>
                                </>
                              ) : (
                                <span className="text-xs font-bold text-slate-300 italic">Unmapped</span>
                              )}
                            </div>
                          </td>
                          <td className="px-10 py-7">
                             <div className={`inline-flex items-center px-4 py-1.5 rounded-full text-[10px] font-black uppercase tracking-[0.15em] border ${
                               synonym.is_active ? 'bg-emerald-50 text-emerald-600 border-emerald-100' : 'bg-slate-50 text-slate-300 border-slate-100'
                             }`}>
                                {synonym.is_active ? 'Active Sync' : 'Offline'}
                             </div>
                          </td>
                          <td className="px-10 py-7 text-right">
                            <div className="flex items-center justify-end gap-3 opacity-0 group-hover:opacity-100 transition-all">
                              <button onClick={() => handleSynonymEdit(synonym)} className="p-3 bg-white rounded-xl text-slate-400 hover:text-indigo-600 border border-slate-100 hover:border-indigo-100 hover:shadow-xl hover:shadow-indigo-100/20 transition-all">
                                <EditIcon className="w-4 h-4" />
                              </button>
                              <button onClick={() => handleSynonymDelete(synonym)} className="p-3 bg-white rounded-xl text-slate-400 hover:text-rose-600 border border-slate-100 hover:border-rose-100 hover:shadow-xl hover:shadow-rose-100/20 transition-all">
                                <TrashIcon className="w-4 h-4" />
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
      <AnimatePresence>
        {showForm && (
          <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-lg flex items-center justify-center p-6 z-[100]">
            <motion.div
              initial={{ opacity: 0, scale: 0.9, y: 30 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: 30 }}
              className="bg-white rounded-[56px] shadow-2xl max-w-4xl w-full border border-white overflow-hidden max-h-[90vh] flex flex-col"
            >
              <div className="p-10 border-b border-slate-50 flex items-center justify-between">
                <div>
                  <h2 className="text-3xl font-black text-slate-900 tracking-tight thai-text uppercase">
                    {editingSynonym ? 'Configure Synonym' : 'Provision New Group'}
                  </h2>
                  <p className="text-sm font-bold text-indigo-500 uppercase tracking-widest mt-1">Rule Definition Interface</p>
                </div>
                <button onClick={() => setShowForm(false)} className="p-5 text-slate-300 hover:text-slate-600 hover:bg-slate-50 rounded-3xl transition-all">
                  <XIcon className="w-8 h-8" />
                </button>
              </div>

              <div className="flex-1 overflow-y-auto p-10 space-y-10 custom-scrollbar">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                  <div className="space-y-4">
                    <label className="block text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] px-1">Identifier Name *</label>
                    <input
                      type="text" required
                      value={formData.name}
                      onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                      className="input-premium w-full text-base font-black thai-text"
                      placeholder="e.g. สมาร์ทโฟน"
                    />
                  </div>
                  <div className="space-y-4">
                    <label className="block text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] px-1">Taxonomy Mapping</label>
                    <select
                      value={formData.category_id}
                      onChange={(e) => setFormData(prev => ({ ...prev, category_id: e.target.value }))}
                      className="select-premium w-full text-base font-black thai-text"
                    >
                      <option value="">Select Anchor Category</option>
                      {categories.map(c => <option key={c.id} value={c.id}>{c.name_th}</option>)}
                    </select>
                  </div>
                </div>

                <div className="space-y-4">
                  <label className="block text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] px-1">Administrative Note</label>
                  <textarea
                    value={formData.description}
                    onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
                    className="input-premium w-full h-32 text-sm font-bold thai-text resize-none"
                    placeholder="Describe the logic or context for this synonym group..."
                  />
                </div>

                <div className="space-y-6">
                  <div className="flex items-center justify-between border-b border-slate-50 pb-4">
                    <label className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em]">Expansion Terms (Variations)</label>
                    <button type="button" onClick={addTerm} className="btn-premium-secondary py-2 px-6">
                      <PlusIcon className="w-4 h-4 mr-2" />
                      Add Variant
                    </button>
                  </div>
                  
                  <div className="grid grid-cols-1 gap-4">
                    {formData.terms.map((term, index) => (
                      <div key={index} className="flex gap-4 items-center p-6 bg-slate-50/50 rounded-3xl border border-slate-100 group/term hover:bg-white hover:border-indigo-100 transition-all">
                        <div className="p-3 bg-white rounded-2xl text-slate-400 border border-slate-100 group-hover/term:text-indigo-600 transition-colors">
                           <LanguagesIcon className="w-5 h-5" />
                        </div>
                        <input
                          type="text"
                          value={term.term}
                          onChange={(e) => {
                             const newTerms = [...formData.terms];
                             newTerms[index].term = e.target.value;
                             setFormData(p => ({ ...p, terms: newTerms }));
                          }}
                          placeholder="Term Variation"
                          className="flex-1 bg-transparent border-none focus:ring-0 text-base font-black text-slate-800 thai-text"
                        />
                        <div className="flex items-center gap-6 pr-2">
                           <label className="flex items-center gap-3 cursor-pointer group/check">
                              <div className="relative">
                                 <input
                                   type="checkbox"
                                   checked={term.is_primary}
                                   onChange={(e) => {
                                      const newTerms = formData.terms.map((t, i) => ({ ...t, is_primary: i === index ? e.target.checked : (e.target.checked ? false : t.is_primary) }));
                                      setFormData(p => ({ ...p, terms: newTerms }));
                                   }}
                                   className="sr-only peer"
                                 />
                                 <div className="w-10 h-5 bg-slate-200 rounded-full peer peer-checked:bg-indigo-600 transition-all after:content-[''] after:absolute after:top-1 after:left-1 after:bg-white after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:after:translate-x-5" />
                              </div>
                              <span className="text-[9px] font-black text-slate-400 uppercase tracking-widest group-hover/check:text-indigo-600 transition-colors">Primary</span>
                           </label>
                           <button onClick={() => setFormData(p => ({ ...p, terms: p.terms.filter((_, i) => i !== index) }))} className="p-2 text-slate-300 hover:text-rose-500 transition-colors">
                              <XIcon className="w-5 h-5" />
                           </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="p-10 bg-slate-50/50 border-t border-slate-100 flex gap-6">
                <button onClick={() => setShowForm(false)} className="btn-premium-secondary flex-1 py-5 h-auto">Discard Config</button>
                <button onClick={handleFormSubmit} className="btn-premium flex-1 py-5 h-auto justify-center group">
                  <SaveIcon className="w-6 h-6 mr-3 group-hover:scale-110 transition-transform" />
                  <span className="thai-text text-lg">Commit Infrastructure</span>
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  )
}

