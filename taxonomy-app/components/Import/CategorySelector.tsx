'use client'

import { useState, useEffect, useRef } from 'react'
import { Search, ChevronDown, Check, X, Loader2 } from 'lucide-react'
import { DatabaseService, TaxonomyNode } from '@/utils/supabase'

interface CategorySelectorProps {
  initialValue?: { id: string; name_th: string; code: string }
  onSelect: (category: TaxonomyNode) => void
  onCancel: () => void
}

export default function CategorySelector({ initialValue, onSelect, onCancel }: CategorySelectorProps) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<TaxonomyNode[]>([])
  const [loading, setLoading] = useState(false)
  const [isOpen, setIsOpen] = useState(true)
  const dropdownRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const searchCategories = async () => {
      if (query.length < 2) {
        setResults([])
        return
      }

      setLoading(true)
      try {
        const nodes = await DatabaseService.searchTaxonomyNodes(query)
        setResults(nodes)
      } catch (error) {
        console.error('Search error:', error)
      } finally {
        setLoading(false)
      }
    }

    const timer = setTimeout(searchCategories, 300)
    return () => clearTimeout(timer)
  }, [query])

  return (
    <div className="relative w-full" ref={dropdownRef}>
      <div className="flex items-center space-x-2 bg-white border rounded-lg p-2 shadow-sm">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            autoFocus
            type="text"
            className="w-full pl-10 pr-4 py-2 text-sm border-none focus:ring-0"
            placeholder="ค้นหาหมวดหมู่ (ชื่อ หรือ รหัส)..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <button
          onClick={onCancel}
          className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {isOpen && (query.length >= 2 || loading) && (
        <div className="absolute z-50 w-full mt-1 bg-white border rounded-lg shadow-xl max-h-60 overflow-y-auto">
          {loading ? (
            <div className="flex items-center justify-center p-4">
              <Loader2 className="w-6 h-6 animate-spin text-blue-500" />
            </div>
          ) : results.length > 0 ? (
            <ul className="py-1">
              {results.map((node) => (
                <li
                  key={node.id}
                  className="px-4 py-2 hover:bg-blue-50 cursor-pointer flex items-center justify-between group"
                  onClick={() => onSelect(node)}
                >
                  <div>
                    <div className="text-sm font-medium text-gray-900">{node.name_th}</div>
                    <div className="text-xs text-gray-500">{node.code}</div>
                  </div>
                  <Check className="w-4 h-4 text-blue-500 opacity-0 group-hover:opacity-100 transition-opacity" />
                </li>
              ))}
            </ul>
          ) : (
            <div className="p-4 text-center text-sm text-gray-500 font-noto-sans-thai">
              ไม่พบหมวดหมู่ที่ตรงกับ "{query}"
            </div>
          )}
        </div>
      )}

      {initialValue && query.length < 2 && !loading && (
        <div className="mt-2 text-xs text-gray-500 font-noto-sans-thai px-2">
          ค่าปัจจุบัน: <span className="font-semibold text-blue-600">{initialValue.name_th} ({initialValue.code})</span>
        </div>
      )}
    </div>
  )
}
