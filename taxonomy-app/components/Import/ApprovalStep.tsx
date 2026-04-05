'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { toast } from 'react-hot-toast'
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
  RefreshCw
} from 'lucide-react'

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
              className="bg-white border border-gray-200 rounded-lg shadow-sm overflow-hidden"
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
                      <h3 className="text-lg font-medium text-gray-900 truncate">
                        {suggestion.product_name}
                      </h3>
                      <div className="flex items-center space-x-2">
                        <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded-full">
                          {Math.round(suggestion.confidence_score * 100)}%
                        </span>
                        <Clock className="w-4 h-4 text-gray-400" />
                      </div>
                    </div>
                    
                    <div className="mt-1 flex items-center space-x-4 text-sm text-gray-600">
                      <span>หมวดหมู่: {suggestion.suggested_category.name_th}</span>
                      <span>•</span>
                      <span>Tokens: {suggestion.tokens.length}</span>
                      {suggestion.units.length > 0 && (
                        <>
                          <span>•</span>
                          <span>Units: {suggestion.units.length}</span>
                        </>
                      )}
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
                      {/* Cleaned Name */}
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          ชื่อที่ทำความสะอาดแล้ว
                        </label>
                        <p className="text-sm text-gray-600 bg-white p-2 rounded border">
                          {suggestion.cleaned_name}
                        </p>
                      </div>

                      {/* Tokens */}
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          Tokens ({suggestion.tokens.length})
                        </label>
                        <div className="flex flex-wrap gap-1">
                          {suggestion.tokens.map((token, idx) => (
                            <span
                              key={idx}
                              className="px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded"
                            >
                              {token}
                            </span>
                          ))}
                        </div>
                      </div>

                      {/* Units */}
                      {suggestion.units.length > 0 && (
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">
                            หน่วย ({suggestion.units.length})
                          </label>
                          <div className="flex flex-wrap gap-1">
                            {suggestion.units.map((unit, idx) => (
                              <span
                                key={idx}
                                className="px-2 py-1 text-xs bg-green-100 text-green-800 rounded"
                              >
                                {unit}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Attributes */}
                      {Object.keys(suggestion.attributes).length > 0 && (
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">
                            คุณสมบัติที่สกัดได้ ({Object.keys(suggestion.attributes).length})
                          </label>
                          <div className="bg-white border rounded p-3 space-y-2">
                            {Object.entries(suggestion.attributes).map(([key, value]) => (
                              <div key={key} className="flex justify-between items-start">
                                <span className="font-medium text-gray-700 capitalize">
                                  {key.replace(/_/g, ' ')}:
                                </span>
                                {renderAttributeValue(value)}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Explanation */}
                      {suggestion.explanation && (
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">
                            คำอธิบาย
                          </label>
                          <p className="text-sm text-gray-600 bg-white p-2 rounded border">
                            {suggestion.explanation}
                          </p>
                        </div>
                      )}

                      {/* Individual Actions */}
                      <div className="flex items-center justify-end space-x-2 pt-2 border-t">
                        <button
                          onClick={() => {
                            setSelectedIds(new Set([suggestion.id]))
                            handleBatchAction('reject')
                          }}
                          disabled={processing}
                          className="px-3 py-1 text-sm text-red-600 hover:text-red-800 transition-colors"
                        >
                          ปฏิเสธ
                        </button>
                        <button
                          onClick={() => {
                            setSelectedIds(new Set([suggestion.id]))
                            handleBatchAction('approve')
                          }}
                          disabled={processing}
                          className="px-4 py-1 text-sm bg-green-500 text-white rounded hover:bg-green-600 transition-colors"
                        >
                          อนุมัติ
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
        <div className="flex justify-center">
          <button
            onClick={() => loadPendingSuggestions(pagination.offset + pagination.limit)}
            disabled={loading}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 transition-colors"
          >
            โหลดเพิ่มเติม
          </button>
        </div>
      )}

      {/* Navigation */}
      <div className="flex justify-between pt-6 border-t">
        <button
          onClick={onBack}
          className="px-4 py-2 text-gray-600 hover:text-gray-800 transition-colors"
        >
          ย้อนกลับ
        </button>
        
        <div className="text-sm text-gray-500">
          แสดง {suggestions.length} จาก {pagination.total} รายการ
        </div>
      </div>
    </div>
  )
}
