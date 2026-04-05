'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { toast } from 'react-hot-toast'
import { 
  PlusIcon,
  EditIcon,
  TrashIcon,
  SearchIcon,
  DownloadIcon,
  UploadIcon,
  TagIcon,
  LinkIcon,
  CheckCircleIcon,
  XCircleIcon,
  FilterIcon,
  SortAscIcon,
  SortDescIcon
} from 'lucide-react'
import { Synonym, SynonymTerm, TaxonomyNode } from '@/utils/supabase'

interface EnhancedSynonymManagerProps {
  synonyms: Synonym[]
  categories: TaxonomyNode[]
  onSynonymCreate?: (synonym: Partial<Synonym>) => void
  onSynonymUpdate?: (id: string, updates: Partial<Synonym>) => void
  onSynonymDelete?: (id: string) => void
  onTermCreate?: (synonymId: string, term: Partial<SynonymTerm>) => void
  onTermUpdate?: (id: string, updates: Partial<SynonymTerm>) => void
  onTermDelete?: (id: string) => void
  onCategoryMap?: (synonymId: string, categoryId: string, weight?: number) => void
  onBulkImport?: (csvData: string) => void
  className?: string
}

interface SynonymFormData {
  name: string
  description: string
  category_id?: string
}

interface TermFormData {
  term: string
  is_primary: boolean
  confidence_score: number
  source: 'manual' | 'auto' | 'imported' | 'ml'
  language: string
}

// CSV Export/Import utilities
const exportToCSV = (synonyms: Synonym[]): string => {
  const headers = [
    'Lemma ID',
    'Lemma Name', 
    'Description',
    'Category',
    'Term',
    'Is Primary',
    'Confidence Score',
    'Source',
    'Language',
    'Verified'
  ]

  const rows = synonyms.flatMap(synonym => 
    synonym.terms?.map(term => [
      synonym.id,
      synonym.name,
      synonym.description || '',
      synonym.category?.name_th || '',
      term.term,
      term.is_primary ? 'Yes' : 'No',
      term.confidence_score,
      term.source,
      term.language,
      term.is_verified ? 'Yes' : 'No'
    ]) || []
  )

  return [headers, ...rows].map(row => 
    row.map(cell => `"${String(cell).replace(/"/g, '""')}"`).join(',')
  ).join('\n')
}

const parseCSV = (csvText: string): Array<{
  lemmaName: string
  description: string
  category: string
  term: string
  isPrimary: boolean
  confidenceScore: number
  source: string
  language: string
  verified: boolean
}> => {
  const lines = csvText.trim().split('\n')
  const headers = lines[0].split(',').map(h => h.replace(/"/g, ''))
  
  return lines.slice(1).map(line => {
    const values = line.match(/(".*?"|[^,]+)(?=\s*,|\s*$)/g)?.map(v => v.replace(/^"|"$/g, '')) || []
    
    return {
      lemmaName: values[1] || '',
      description: values[2] || '',
      category: values[3] || '',
      term: values[4] || '',
      isPrimary: values[5] === 'Yes',
      confidenceScore: parseFloat(values[6]) || 1.0,
      source: values[7] || 'manual',
      language: values[8] || 'th',
      verified: values[9] === 'Yes'
    }
  })
}

export default function EnhancedSynonymManager({
  synonyms,
  categories,
  onSynonymCreate,
  onSynonymUpdate,
  onSynonymDelete,
  onTermCreate,
  onTermUpdate,
  onTermDelete,
  onCategoryMap,
  onBulkImport,
  className = ''
}: EnhancedSynonymManagerProps) {
  const [selectedSynonym, setSelectedSynonym] = useState<Synonym | null>(null)
  const [showSynonymForm, setShowSynonymForm] = useState(false)
  const [showTermForm, setShowTermForm] = useState(false)
  const [showImportModal, setShowImportModal] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [filterCategory, setFilterCategory] = useState<string>('')
  const [filterSource, setFilterSource] = useState<string>('')
  const [sortBy, setSortBy] = useState<'name' | 'terms' | 'category'>('name')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc')
  const [importCsv, setImportCsv] = useState('')

  const [synonymForm, setSynonymForm] = useState<SynonymFormData>({
    name: '',
    description: '',
    category_id: ''
  })

  const [termForm, setTermForm] = useState<TermFormData>({
    term: '',
    is_primary: false,
    confidence_score: 1.0,
    source: 'manual',
    language: 'th'
  })

  // Filter and sort synonyms
  const filteredSynonyms = synonyms
    .filter(synonym => {
      const matchesSearch = !searchTerm || 
        synonym.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        synonym.description?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        synonym.terms?.some(term => term.term.toLowerCase().includes(searchTerm.toLowerCase()))

      const matchesCategory = !filterCategory || synonym.category_id === filterCategory
      
      const matchesSource = !filterSource || 
        synonym.terms?.some(term => term.source === filterSource)

      return matchesSearch && matchesCategory && matchesSource
    })
    .sort((a, b) => {
      let aValue: any, bValue: any

      switch (sortBy) {
        case 'name':
          aValue = a.name
          bValue = b.name
          break
        case 'terms':
          aValue = a.terms?.length || 0
          bValue = b.terms?.length || 0
          break
        case 'category':
          aValue = a.category?.name_th || ''
          bValue = b.category?.name_th || ''
          break
        default:
          return 0
      }

      if (sortOrder === 'asc') {
        return aValue > bValue ? 1 : -1
      } else {
        return aValue < bValue ? 1 : -1
      }
    })

  const handleSynonymSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    
    if (selectedSynonym) {
      onSynonymUpdate?.(selectedSynonym.id, synonymForm)
    } else {
      onSynonymCreate?.(synonymForm)
    }
    
    setShowSynonymForm(false)
    setSynonymForm({ name: '', description: '', category_id: '' })
    setSelectedSynonym(null)
  }

  const handleTermSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!selectedSynonym) return
    
    onTermCreate?.(selectedSynonym.id, termForm)
    setShowTermForm(false)
    setTermForm({
      term: '',
      is_primary: false,
      confidence_score: 1.0,
      source: 'manual',
      language: 'th'
    })
  }

  const handleExport = () => {
    const csv = exportToCSV(synonyms)
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `synonyms-export-${new Date().toISOString().split('T')[0]}.csv`
    a.click()
    URL.revokeObjectURL(url)
    toast.success('ส่งออก CSV สำเร็จ')
  }

  const handleImport = () => {
    if (!importCsv.trim()) {
      toast.error('กรุณาใส่ข้อมูล CSV')
      return
    }
    
    try {
      onBulkImport?.(importCsv)
      setShowImportModal(false)
      setImportCsv('')
      toast.success('นำเข้าข้อมูลสำเร็จ')
    } catch (error) {
      toast.error('เกิดข้อผิดพลาดในการนำเข้าข้อมูล')
    }
  }

  const toggleSort = (field: typeof sortBy) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')
    } else {
      setSortBy(field)
      setSortOrder('asc')
    }
  }

  return (
    <div className={`bg-white rounded-lg border border-gray-200 ${className}`}>
      {/* Header */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900">จัดการ Synonym</h2>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowImportModal(true)}
              className="btn-secondary"
            >
              <UploadIcon className="w-4 h-4 mr-2" />
              นำเข้า CSV
            </button>
            
            <button
              onClick={handleExport}
              className="btn-secondary"
            >
              <DownloadIcon className="w-4 h-4 mr-2" />
              ส่งออก CSV
            </button>
            
            <button
              onClick={() => {
                setSelectedSynonym(null)
                setSynonymForm({ name: '', description: '', category_id: '' })
                setShowSynonymForm(true)
              }}
              className="btn-premium"
            >
              <PlusIcon className="w-4 h-4 mr-2" />
              เพิ่ม Lemma
            </button>
          </div>
        </div>

        {/* Filters */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="relative">
            <SearchIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="ค้นหา lemma หรือ terms..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="input-premium pl-10"
            />
          </div>

          <select
            value={filterCategory}
            onChange={(e) => setFilterCategory(e.target.value)}
            className="input-premium"
          >
            <option value="">ทุกหมวดหมู่</option>
            {categories.map(cat => (
              <option key={cat.id} value={cat.id}>{cat.name_th}</option>
            ))}
          </select>

          <select
            value={filterSource}
            onChange={(e) => setFilterSource(e.target.value)}
            className="input-premium"
          >
            <option value="">ทุกแหล่งที่มา</option>
            <option value="manual">Manual</option>
            <option value="auto">Auto</option>
            <option value="imported">Imported</option>
            <option value="ml">ML</option>
          </select>

          <div className="flex items-center space-x-2">
            <FilterIcon className="w-4 h-4 text-gray-400" />
            <span className="text-sm text-gray-600">
              {filteredSynonyms.length} รายการ
            </span>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex h-96">
        {/* Synonym List */}
        <div className="w-1/2 border-r border-gray-200">
          <div className="p-4 border-b border-gray-200 bg-gray-50">
            <div className="flex items-center justify-between">
              <h3 className="font-medium text-gray-900">Lemma Groups</h3>
              <div className="flex items-center space-x-1">
                <button
                  onClick={() => toggleSort('name')}
                  className={`p-1 rounded ${sortBy === 'name' ? 'bg-blue-100 text-blue-600' : 'text-gray-400'}`}
                >
                  {sortOrder === 'asc' ? <SortAscIcon className="w-4 h-4" /> : <SortDescIcon className="w-4 h-4" />}
                </button>
              </div>
            </div>
          </div>

          <div className="overflow-y-auto h-full">
            {filteredSynonyms.map((synonym) => (
              <motion.div
                key={synonym.id}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className={`
                  p-4 border-b border-gray-100 cursor-pointer transition-colors
                  ${selectedSynonym?.id === synonym.id ? 'bg-blue-50 border-blue-200' : 'hover:bg-gray-50'}
                `}
                onClick={() => setSelectedSynonym(synonym)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center space-x-2">
                      <TagIcon className="w-4 h-4 text-blue-500 flex-shrink-0" />
                      <h4 className="font-medium text-gray-900 truncate">
                        {synonym.name}
                      </h4>
                    </div>
                    
                    {synonym.description && (
                      <p className="text-sm text-gray-600 mt-1 truncate">
                        {synonym.description}
                      </p>
                    )}
                    
                    <div className="flex items-center space-x-4 mt-2">
                      <span className="text-xs text-gray-500">
                        {synonym.terms?.length || 0} terms
                      </span>
                      
                      {synonym.category && (
                        <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">
                          {synonym.category.name_th}
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center space-x-1 ml-2">
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        setSelectedSynonym(synonym)
                        setSynonymForm({
                          name: synonym.name,
                          description: synonym.description || '',
                          category_id: synonym.category_id || ''
                        })
                        setShowSynonymForm(true)
                      }}
                      className="p-1 rounded hover:bg-yellow-100 text-yellow-600"
                    >
                      <EditIcon className="w-4 h-4" />
                    </button>
                    
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        onSynonymDelete?.(synonym.id)
                      }}
                      className="p-1 rounded hover:bg-red-100 text-red-600"
                    >
                      <TrashIcon className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Term Details */}
        <div className="w-1/2">
          {selectedSynonym ? (
            <div className="h-full flex flex-col">
              <div className="p-4 border-b border-gray-200 bg-gray-50">
                <div className="flex items-center justify-between">
                  <h3 className="font-medium text-gray-900">
                    Terms for "{selectedSynonym.name}"
                  </h3>
                  <button
                    onClick={() => setShowTermForm(true)}
                    className="btn-premium text-sm"
                  >
                    <PlusIcon className="w-4 h-4 mr-1" />
                    เพิ่ม Term
                  </button>
                </div>
              </div>

              <div className="flex-1 overflow-y-auto p-4">
                {selectedSynonym.terms?.map((term) => (
                  <motion.div
                    key={term.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="p-3 border border-gray-200 rounded-lg mb-3"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2">
                          <span className="font-medium text-gray-900">
                            {term.term}
                          </span>
                          
                          {term.is_primary && (
                            <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded-full">
                              Primary
                            </span>
                          )}
                          
                          <span className={`text-xs px-2 py-1 rounded-full ${
                            term.is_verified 
                              ? 'bg-green-100 text-green-800' 
                              : 'bg-yellow-100 text-yellow-800'
                          }`}>
                            {term.is_verified ? 'Verified' : 'Pending'}
                          </span>
                        </div>
                        
                        <div className="flex items-center space-x-4 mt-2 text-sm text-gray-600">
                          <span>Score: {(term.confidence_score * 100).toFixed(0)}%</span>
                          <span>Source: {term.source}</span>
                          <span>Lang: {term.language}</span>
                        </div>
                      </div>

                      <div className="flex items-center space-x-1 ml-2">
                        <button
                          onClick={() => onTermUpdate?.(term.id, { is_verified: !term.is_verified })}
                          className={`p-1 rounded ${
                            term.is_verified 
                              ? 'hover:bg-yellow-100 text-yellow-600' 
                              : 'hover:bg-green-100 text-green-600'
                          }`}
                        >
                          {term.is_verified ? <XCircleIcon className="w-4 h-4" /> : <CheckCircleIcon className="w-4 h-4" />}
                        </button>
                        
                        <button
                          onClick={() => onTermDelete?.(term.id)}
                          className="p-1 rounded hover:bg-red-100 text-red-600"
                        >
                          <TrashIcon className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          ) : (
            <div className="h-full flex items-center justify-center text-gray-500">
              <div className="text-center">
                <TagIcon className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                <p>เลือก Lemma เพื่อดู Terms</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Synonym Form Modal */}
      {showSynonymForm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md mx-4">
            <h3 className="text-lg font-semibold mb-4">
              {selectedSynonym ? 'แก้ไข Lemma' : 'เพิ่ม Lemma ใหม่'}
            </h3>
            
            <form onSubmit={handleSynonymSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  ชื่อ Lemma
                </label>
                <input
                  type="text"
                  value={synonymForm.name}
                  onChange={(e) => setSynonymForm(prev => ({ ...prev, name: e.target.value }))}
                  className="input-premium"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  คำอธิบาย
                </label>
                <textarea
                  value={synonymForm.description}
                  onChange={(e) => setSynonymForm(prev => ({ ...prev, description: e.target.value }))}
                  className="input-premium"
                  rows={3}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  หมวดหมู่
                </label>
                <select
                  value={synonymForm.category_id}
                  onChange={(e) => setSynonymForm(prev => ({ ...prev, category_id: e.target.value }))}
                  className="input-premium"
                >
                  <option value="">ไม่ระบุหมวดหมู่</option>
                  {categories.map(cat => (
                    <option key={cat.id} value={cat.id}>{cat.name_th}</option>
                  ))}
                </select>
              </div>

              <div className="flex justify-end space-x-2">
                <button
                  type="button"
                  onClick={() => setShowSynonymForm(false)}
                  className="btn-secondary"
                >
                  ยกเลิก
                </button>
                <button type="submit" className="btn-premium">
                  {selectedSynonym ? 'อัปเดต' : 'สร้าง'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Term Form Modal */}
      {showTermForm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md mx-4">
            <h3 className="text-lg font-semibold mb-4">เพิ่ม Term ใหม่</h3>
            
            <form onSubmit={handleTermSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  คำ/วลี
                </label>
                <input
                  type="text"
                  value={termForm.term}
                  onChange={(e) => setTermForm(prev => ({ ...prev, term: e.target.value }))}
                  className="input-premium"
                  required
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    ภาษา
                  </label>
                  <select
                    value={termForm.language}
                    onChange={(e) => setTermForm(prev => ({ ...prev, language: e.target.value }))}
                    className="input-premium"
                  >
                    <option value="th">ไทย</option>
                    <option value="en">อังกฤษ</option>
                    <option value="mixed">ผสม</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    แหล่งที่มา
                  </label>
                  <select
                    value={termForm.source}
                    onChange={(e) => setTermForm(prev => ({ ...prev, source: e.target.value as any }))}
                    className="input-premium"
                  >
                    <option value="manual">Manual</option>
                    <option value="auto">Auto</option>
                    <option value="imported">Imported</option>
                    <option value="ml">ML</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  คะแนนความเชื่อมั่น
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={termForm.confidence_score}
                  onChange={(e) => setTermForm(prev => ({ ...prev, confidence_score: parseFloat(e.target.value) }))}
                  className="w-full"
                />
                <div className="text-sm text-gray-600 text-center">
                  {(termForm.confidence_score * 100).toFixed(0)}%
                </div>
              </div>

              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="is_primary"
                  checked={termForm.is_primary}
                  onChange={(e) => setTermForm(prev => ({ ...prev, is_primary: e.target.checked }))}
                  className="rounded"
                />
                <label htmlFor="is_primary" className="ml-2 text-sm text-gray-700">
                  เป็น Primary Term
                </label>
              </div>

              <div className="flex justify-end space-x-2">
                <button
                  type="button"
                  onClick={() => setShowTermForm(false)}
                  className="btn-secondary"
                >
                  ยกเลิก
                </button>
                <button type="submit" className="btn-premium">
                  เพิ่ม Term
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Import Modal */}
      {showImportModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-2xl mx-4">
            <h3 className="text-lg font-semibold mb-4">นำเข้าข้อมูล CSV</h3>
            
            <div className="mb-4 text-sm text-gray-600">
              <p>รูปแบบ CSV: Lemma ID, Lemma Name, Description, Category, Term, Is Primary, Confidence Score, Source, Language, Verified</p>
            </div>
            
            <textarea
              value={importCsv}
              onChange={(e) => setImportCsv(e.target.value)}
              placeholder="วางข้อมูล CSV ที่นี่..."
              className="w-full h-64 p-3 border border-gray-300 rounded-lg font-mono text-sm"
            />
            
            <div className="flex justify-end space-x-2 mt-4">
              <button
                onClick={() => setShowImportModal(false)}
                className="btn-secondary"
              >
                ยกเลิก
              </button>
              <button
                onClick={handleImport}
                className="btn-premium"
              >
                นำเข้าข้อมูล
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
