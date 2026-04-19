'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { toast } from 'react-hot-toast'
import Link from 'next/link'
import { 
  CheckCircle,
  XCircle,
  Clock,
  Eye,
  ChevronDown,
  ChevronUp,
  Play,
  Pause,
  Check,
  X,
  RefreshCw,
  Edit2,
  AlertTriangle,
  ExternalLink
} from 'lucide-react'
import CategorySelector from './CategorySelector'
import { TaxonomyNode } from '@/utils/supabase'

interface PendingSuggestion {
  id: string
  product_name: string
  cleaned_name: string
  tokens: string[]
  units: string[]
  attributes: Record<string, any>
  suggested_category: {
    id: string
    name_th: string
    code: string
  }
  confidence_score: number
  explanation: string
  created_at: string
  status: 'pending' | 'approved' | 'rejected'
  metadata: {
    is_duplicate_detected?: boolean
    potential_duplicates?: Array<{
      id: string
      name_th: string
      similarity: number
    }>
  }
}

interface ApprovalStepProps {
  onComplete?: (results: any) => void
  onBack?: () => void
}

export default function ApprovalStep({ onComplete, onBack }: ApprovalStepProps) {
  const [suggestions, setSuggestions] = useState<PendingSuggestion[]>([])
  const [loading, setLoading] = useState(true)
  const [processing, setProcessing] = useState(false)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set())
  const [editingId, setEditingId] = useState<string | null>(null)
  const [pagination, setPagination] = useState({
    total: 0,
    limit: 20,
    offset: 0,
    has_more: false
  })

  // Load pending suggestions
  const loadPendingSuggestions = async (offset = 0) => {
    try {
      setLoading(true)
      const response = await fetch(`/api/import/pending?limit=${pagination.limit}&offset=${offset}`)
      
      if (!response.ok) {
        throw new Error('Failed to load pending suggestions')
      }

      const data = await response.json()
      
      if (data.success) {
        setSuggestions(data.data)
        setPagination(data.pagination)
      } else {
        throw new Error(data.error || 'Unknown error')
      }
    } catch (error) {
      console.error('Error loading suggestions:', error)
      toast.error('ไม่สามารถโหลดข้อมูลได้')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadPendingSuggestions()
  }, [])

  // Handle manual category change
  const handleCategoryChange = async (suggestionId: string, newCategory: TaxonomyNode) => {
    try {
      setProcessing(true)
      const response = await fetch('/api/import/pending', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          suggestion_id: suggestionId,
          new_category_id: newCategory.id
        })
      })

      if (!response.ok) throw new Error('Failed to update category')

      toast.success('อัปเดตหมวดหมู่เรียบร้อยแล้ว')
      setEditingId(null)
      
      // Update local state
      setSuggestions(prev => prev.map(s => 
        s.id === suggestionId 
          ? { 
              ...s, 
              suggested_category: { 
                id: newCategory.id, 
                name_th: newCategory.name_th, 
                code: newCategory.code 
              },
              confidence_score: 1.0 // Manual override sets high confidence
            } 
          : s
      ))
    } catch (error) {
      toast.error('ไม่สามารถเปลี่ยนหมวดหมู่ได้')
    } finally {
      setProcessing(false)
    }
  }

  // Toggle selection
  const toggleSelection = (id: string) => {
    const newSelected = new Set(selectedIds)
    if (newSelected.has(id)) {
      newSelected.delete(id)
    } else {
      newSelected.add(id)
    }
    setSelectedIds(newSelected)
  }

  // Select all/none
  const toggleSelectAll = () => {
    if (selectedIds.size === suggestions.length) {
      setSelectedIds(new Set())
    } else {
      setSelectedIds(new Set(suggestions.map(s => s.id)))
    }
  }

  // Toggle expanded view
  const toggleExpanded = (id: string) => {
    const newExpanded = new Set(expandedIds)
    if (newExpanded.has(id)) {
      newExpanded.delete(id)
    } else {
      newExpanded.add(id)
    }
    setExpandedIds(newExpanded)
  }

  // Batch approve/reject
  const handleBatchAction = async (action: 'approve' | 'reject') => {
    if (selectedIds.size === 0) {
      toast.error('กรุณาเลือกรายการที่ต้องการดำเนินการ')
      return
    }

    try {
      setProcessing(true)
      
      const response = await fetch('/api/import/pending', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action,
          suggestion_ids: Array.from(selectedIds)
        })
      })

      if (!response.ok) {
        throw new Error('Failed to process batch action')
      }

      const result = await response.json()
      
      if (result.success) {
        toast.success(`${action === 'approve' ? 'อนุมัติ' : 'ปฏิเสธ'} ${result.results.success} รายการสำเร็จ`)
        
        if (result.results.failed > 0) {
          toast.error(`ไม่สามารถดำเนินการ ${result.results.failed} รายการได้`)
        }

        // Reload data
        await loadPendingSuggestions(pagination.offset)
        setSelectedIds(new Set())

        // Call completion callback if all approved
        if (action === 'approve' && onComplete) {
          onComplete(result)
        }
      } else {
        throw new Error(result.error || 'Unknown error')
      }
    } catch (error) {
      console.error('Batch action error:', error)
      toast.error('เกิดข้อผิดพลาดในการดำเนินการ')
    } finally {
      setProcessing(false)
    }
  }

  // Render attribute value
  const renderAttributeValue = (value: any) => {
    if (typeof value === 'object' && value.matches) {
      return (
        <div className="text-sm">
          <div className="font-medium text-gray-700">Matches:</div>
          <div className="text-gray-600">{value.matches.join(', ')}</div>
          <div className="text-xs text-gray-500">Rule: {value.rule_code}</div>
        </div>
      )
    }
    return <span className="text-gray-600">{String(value)}</span>
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="flex items-center space-x-3">
          <RefreshCw className="w-6 h-6 animate-spin text-blue-500" />
          <span className="text-gray-600">กำลังโหลดข้อมูล...</span>
        </div>
      </div>
    )
  }

  if (suggestions.length === 0) {
    return (
      <div className="text-center py-12">
        <CheckCircle className="w-16 h-16 text-green-500 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">ไม่มีรายการรอการอนุมัติ</h3>
        <p className="text-gray-600 mb-6">ทุกรายการได้รับการอนุมัติแล้ว หรือยังไม่มีการประมวลผลใหม่</p>
        <button
          onClick={() => loadPendingSuggestions()}
          className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
        >
          รีเฟรช
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">อนุมัติรายการสินค้า</h2>
          <p className="text-gray-600">รายการที่รอการอนุมัติ: {pagination.total} รายการ</p>
        </div>
        
        <div className="flex items-center space-x-3">
          <button
            onClick={() => loadPendingSuggestions(pagination.offset)}
            disabled={loading}
            className="px-3 py-2 text-gray-600 hover:text-gray-800 transition-colors"
          >
            <RefreshCw className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Batch Actions */}
      <div className="flex items-center justify-between bg-gray-50 p-4 rounded-lg">
        <div className="flex items-center space-x-4">
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={selectedIds.size === suggestions.length && suggestions.length > 0}
              onChange={toggleSelectAll}
              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span className="text-sm text-gray-700">
              เลือกทั้งหมด ({selectedIds.size}/{suggestions.length})
            </span>
          </label>
        </div>

        {selectedIds.size > 0 && (
          <div className="flex items-center space-x-2">
            <button
              onClick={() => handleBatchAction('approve')}
              disabled={processing}
              className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 disabled:opacity-50 transition-colors flex items-center space-x-2"
            >
              <Check className="w-4 h-4" />
              <span>อนุมัติ ({selectedIds.size})</span>
            </button>
            
            <button
              onClick={() => handleBatchAction('reject')}
              disabled={processing}
              className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 disabled:opacity-50 transition-colors flex items-center space-x-2"
            >
              <X className="w-4 h-4" />
              <span>ปฏิเสธ ({selectedIds.size})</span>
            </button>
          </div>
        )}
      </div>

      {/* Suggestions List */}
      <div className="space-y-4">
        <AnimatePresence>
          {suggestions.map((suggestion) => (
            <motion.div
              key={suggestion.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className={`bg-white border rounded-lg shadow-sm overflow-hidden ${
                suggestion.metadata?.is_duplicate_detected ? 'border-amber-200 ring-1 ring-amber-100' : 'border-gray-200'
              }`}
            >
              {/* Main Row */}
              <div className="p-4">
                <div className="flex items-center space-x-4">
                  <input
                    type="checkbox"
                    checked={selectedIds.has(suggestion.id)}
                    onChange={() => toggleSelection(suggestion.id)}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3 truncate">
                        <h3 className="text-lg font-medium text-gray-900 truncate">
                          {suggestion.product_name}
                        </h3>
                        {suggestion.metadata?.is_duplicate_detected && (
                          <div className="flex-shrink-0 flex items-center gap-1 px-2 py-0.5 bg-amber-50 text-amber-700 text-[10px] font-bold rounded-md border border-amber-100">
                            <AlertTriangle className="w-3 h-3" />
                            DUPLICATE
                          </div>
                        )}
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                          suggestion.confidence_score >= 0.8 ? 'bg-green-100 text-green-800' :
                          suggestion.confidence_score >= 0.5 ? 'bg-amber-100 text-amber-800' :
                          'bg-rose-100 text-rose-800'
                        }`}>
                          {Math.round(suggestion.confidence_score * 100)}%
                        </span>
                        <Clock className="w-4 h-4 text-gray-400" />
                      </div>
                    </div>
                    
                    <div className="mt-1 flex items-center justify-between">
                      <div className="flex items-center space-x-4 text-sm text-gray-600">
                        {editingId === suggestion.id ? (
                          <div className="w-full min-w-[300px]">
                            <CategorySelector 
                              initialValue={suggestion.suggested_category}
                              onSelect={(cat) => handleCategoryChange(suggestion.id, cat)}
                              onCancel={() => setEditingId(null)}
                            />
                          </div>
                        ) : (
                          <div className="flex items-center space-x-2 group">
                             <span>หมวดหมู่: <span className="font-semibold text-slate-900">{suggestion.suggested_category?.name_th || 'ไม่ระบุ'}</span></span>
                             <button 
                               onClick={() => setEditingId(suggestion.id)}
                               className="opacity-0 group-hover:opacity-100 p-1 text-blue-500 hover:bg-blue-50 rounded transition-all"
                             >
                               <Edit2 className="w-3.5 h-3.5" />
                             </button>
                          </div>
                        )}
                        {!editingId && (
                          <>
                            <span>•</span>
                            <span>Tokens: {suggestion.tokens?.length || 0}</span>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                  
                  <button
                    onClick={() => toggleExpanded(suggestion.id)}
                    className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
                  >
                    {expandedIds.has(suggestion.id) ? (
                      <ChevronUp className="w-5 h-5" />
                    ) : (
                      <ChevronDown className="w-5 h-5" />
                    )}
                  </button>
                </div>
              </div>

              {/* Expanded Details */}
              <AnimatePresence>
                {expandedIds.has(suggestion.id) && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="border-t border-gray-200 bg-gray-50"
                  >
                    <div className="p-4 space-y-4">
                      {/* Potential Duplicates Warning */}
                      {suggestion.metadata?.is_duplicate_detected && (
                        <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 shadow-inner">
                          <div className="flex items-center gap-2 text-amber-800 font-bold text-sm mb-3">
                            <AlertTriangle className="w-4 h-4" />
                            พบสินค้าที่คล้ายกันในระบบ (Potential Duplicates)
                          </div>
                          <div className="space-y-2">
                            {suggestion.metadata.potential_duplicates?.map((dup, idx) => (
                              <div key={idx} className="flex items-center justify-between bg-white/80 p-2.5 rounded-lg text-xs border border-amber-100">
                                <span className="font-medium text-slate-700">{dup.name_th}</span>
                                <div className="flex items-center gap-4">
                                  <span className="text-amber-600 font-bold">Similarity: {Math.round((1 - (dup as any).similarity) * 100)}%</span>
                                  <Link href={`/products/${dup.id}`} target="_blank" className="text-blue-600 hover:text-blue-800 font-bold flex items-center gap-1 transition-colors">
                                    เปิดดู <ExternalLink className="w-3 h-3" />
                                  </Link>
                                </div>
                              </div>
                            ))}
                          </div>
                          <p className="mt-3 text-[10px] text-amber-600 font-medium">
                            * หากเป็นสินค้าตัวเดียวกัน ควรปฏิเสธ (Reject) เพื่อไม่ให้ข้อมูลซ้ำซ้อน
                          </p>
                        </div>
                      )}

                      {/* Cleaned Name */}
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          ชื่อที่ทำความสะอาดแล้ว
                        </label>
                        <p className="text-sm text-gray-600 bg-white p-2 rounded border border-gray-200">
                          {suggestion.cleaned_name}
                        </p>
                      </div>

                      {/* Tokens & Units */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">
                            Tokens ({suggestion.tokens?.length || 0})
                          </label>
                          <div className="flex flex-wrap gap-1">
                            {suggestion.tokens?.map((token, idx) => (
                              <span key={idx} className="px-2 py-1 text-xs bg-blue-50 text-blue-700 border border-blue-100 rounded">
                                {token}
                              </span>
                            ))}
                          </div>
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">
                            หน่วย ({suggestion.units?.length || 0})
                          </label>
                          <div className="flex flex-wrap gap-1">
                            {suggestion.units?.map((unit, idx) => (
                              <span key={idx} className="px-2 py-1 text-xs bg-green-50 text-green-700 border border-green-100 rounded">
                                {unit}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>

                      {/* Attributes */}
                      {suggestion.attributes && Object.keys(suggestion.attributes).length > 0 && (
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">
                            คุณสมบัติที่สกัดได้ ({Object.keys(suggestion.attributes).length})
                          </label>
                          <div className="bg-white border border-gray-200 rounded-lg p-3 space-y-2">
                            {Object.entries(suggestion.attributes).map(([key, value]) => (
                              <div key={key} className="flex justify-between items-start">
                                <span className="font-semibold text-slate-500 text-xs uppercase tracking-wider capitalize">
                                  {key.replace(/_/g, ' ')}:
                                </span>
                                {renderAttributeValue(value)}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Individual Actions */}
                      <div className="flex items-center justify-end space-x-3 pt-4 border-t border-gray-200">
                        <button
                          onClick={() => {
                            setSelectedIds(new Set([suggestion.id]))
                            handleBatchAction('reject')
                          }}
                          disabled={processing}
                          className="px-4 py-2 text-sm font-bold text-rose-600 hover:bg-rose-50 rounded-xl transition-colors"
                        >
                          ปฏิเสธรายการนี้
                        </button>
                        <button
                          onClick={() => {
                            setSelectedIds(new Set([suggestion.id]))
                            handleBatchAction('approve')
                          }}
                          disabled={processing}
                          className="px-6 py-2 text-sm font-bold bg-emerald-600 text-white rounded-xl hover:bg-emerald-700 shadow-lg shadow-emerald-600/20 transition-all active:scale-95"
                        >
                          อนุมัติเข้าคลัง
                        </button>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Pagination */}
      {pagination.has_more && (
        <div className="flex justify-center mt-10">
          <button
            onClick={() => loadPendingSuggestions(pagination.offset + pagination.limit)}
            disabled={loading}
            className="flex items-center gap-2 px-8 py-3 bg-white border border-slate-200 text-slate-600 rounded-2xl hover:border-blue-300 hover:text-blue-600 disabled:opacity-50 transition-all font-black text-xs uppercase tracking-widest shadow-sm"
          >
            {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : 'Load More Suggestions'}
          </button>
        </div>
      )}

      {/* Navigation */}
      <div className="flex justify-between items-center pt-8 border-t border-gray-200 mt-10">
        <button
          onClick={onBack}
          className="px-6 py-2 text-slate-400 hover:text-slate-900 transition-colors font-bold text-sm flex items-center gap-2"
        >
          ← ย้อนกลับ
        </button>
        
        <div className="text-xs font-black text-slate-300 uppercase tracking-[0.2em]">
          Total {pagination.total} Records Found
        </div>
      </div>
    </div>
  )
}
