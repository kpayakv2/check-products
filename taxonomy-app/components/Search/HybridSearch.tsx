'use client'

import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  SearchIcon,
  FilterIcon,
  SortAscIcon,
  SortDescIcon,
  TagIcon,
  TrendingUpIcon,
  ZapIcon,
  TargetIcon,
  ClockIcon
} from 'lucide-react'
import { Product, TaxonomyNode } from '@/utils/supabase'

interface SearchResult {
  product: Product
  score: number
  matchType: 'vector' | 'text' | 'hybrid'
  matchedTokens: string[]
  explanation: string
}

interface HybridSearchProps {
  onSearch?: (query: string, filters: SearchFilters) => Promise<SearchResult[]>
  categories: TaxonomyNode[]
  className?: string
}

interface SearchFilters {
  categories: string[]
  priceRange: [number, number]
  confidence: [number, number]
  matchType: 'all' | 'vector' | 'text' | 'hybrid'
  sortBy: 'relevance' | 'price' | 'confidence' | 'date'
  sortOrder: 'asc' | 'desc'
}

// Token highlighting utility
const highlightTokens = (text: string, tokens: string[]): JSX.Element => {
  if (!tokens.length) return <span>{text}</span>

  let highlightedText = text
  const highlights: Array<{ start: number; end: number; token: string }> = []

  // Find all token positions
  tokens.forEach(token => {
    const regex = new RegExp(token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi')
    let match
    while ((match = regex.exec(text)) !== null) {
      highlights.push({
        start: match.index,
        end: match.index + match[0].length,
        token: match[0]
      })
    }
  })

  // Sort by position and merge overlapping highlights
  highlights.sort((a, b) => a.start - b.start)
  const mergedHighlights = highlights.reduce((acc, curr) => {
    if (acc.length === 0) return [curr]
    
    const last = acc[acc.length - 1]
    if (curr.start <= last.end) {
      last.end = Math.max(last.end, curr.end)
      return acc
    }
    
    return [...acc, curr]
  }, [] as typeof highlights)

  // Build JSX with highlights
  if (mergedHighlights.length === 0) return <span>{text}</span>

  const parts: JSX.Element[] = []
  let lastEnd = 0

  mergedHighlights.forEach((highlight, index) => {
    // Add text before highlight
    if (highlight.start > lastEnd) {
      parts.push(
        <span key={`text-${index}`}>
          {text.slice(lastEnd, highlight.start)}
        </span>
      )
    }

    // Add highlighted text
    parts.push(
      <mark 
        key={`highlight-${index}`}
        className="bg-yellow-200 px-1 rounded font-medium"
      >
        {text.slice(highlight.start, highlight.end)}
      </mark>
    )

    lastEnd = highlight.end
  })

  // Add remaining text
  if (lastEnd < text.length) {
    parts.push(
      <span key="text-end">
        {text.slice(lastEnd)}
      </span>
    )
  }

  return <span>{parts}</span>
}

// Search suggestions based on recent queries
const getSearchSuggestions = (query: string): string[] => {
  const suggestions = [
    'iPhone 15 Pro Max',
    'Samsung Galaxy S24',
    'MacBook Air M2',
    'iPad Pro',
    'AirPods Pro',
    'เสื้อยืดผ้าฝ้าย',
    'กางเกงยีนส์',
    'รองเท้าผ้าใบ Nike',
    'กระเป๋าเป้หนังแท้',
    'นาฬิกาข้อมือ',
    'ข้าวหอมมะลิ',
    'น้ำมันปาล์ม',
    'กาแฟสำเร็จรูป',
    'ชาเขียว',
    'นมข้นหวาน'
  ]

  if (!query) return suggestions.slice(0, 5)
  
  return suggestions
    .filter(s => s.toLowerCase().includes(query.toLowerCase()))
    .slice(0, 5)
}

export default function HybridSearch({
  onSearch,
  categories,
  className = ''
}: HybridSearchProps) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [showFilters, setShowFilters] = useState(false)
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [searchHistory, setSearchHistory] = useState<string[]>([])
  const searchInputRef = useRef<HTMLInputElement>(null)

  const [filters, setFilters] = useState<SearchFilters>({
    categories: [],
    priceRange: [0, 100000],
    confidence: [0, 1],
    matchType: 'all',
    sortBy: 'relevance',
    sortOrder: 'desc'
  })

  const [searchStats, setSearchStats] = useState({
    totalResults: 0,
    vectorMatches: 0,
    textMatches: 0,
    hybridMatches: 0,
    searchTime: 0
  })

  const handleSearch = async () => {
    if (!query.trim() || !onSearch) return

    setIsSearching(true)
    const startTime = Date.now()

    try {
      const searchResults = await onSearch(query, filters)
      setResults(searchResults)
      
      // Update search stats
      const searchTime = Date.now() - startTime
      setSearchStats({
        totalResults: searchResults.length,
        vectorMatches: searchResults.filter(r => r.matchType === 'vector').length,
        textMatches: searchResults.filter(r => r.matchType === 'text').length,
        hybridMatches: searchResults.filter(r => r.matchType === 'hybrid').length,
        searchTime
      })

      // Add to search history
      if (!searchHistory.includes(query)) {
        setSearchHistory(prev => [query, ...prev.slice(0, 9)])
      }

    } catch (error) {
      console.error('Search error:', error)
    } finally {
      setIsSearching(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch()
      setShowSuggestions(false)
    }
  }

  const getMatchTypeIcon = (type: string) => {
    switch (type) {
      case 'vector':
        return <ZapIcon className="w-4 h-4 text-purple-500" />
      case 'text':
        return <TagIcon className="w-4 h-4 text-blue-500" />
      case 'hybrid':
        return <TargetIcon className="w-4 h-4 text-green-500" />
      default:
        return <SearchIcon className="w-4 h-4 text-gray-500" />
    }
  }

  const getMatchTypeLabel = (type: string) => {
    switch (type) {
      case 'vector':
        return 'Semantic'
      case 'text':
        return 'Keyword'
      case 'hybrid':
        return 'Hybrid'
      default:
        return 'Unknown'
    }
  }

  const suggestions = getSearchSuggestions(query)

  return (
    <div className={`bg-white rounded-lg border border-gray-200 ${className}`}>
      {/* Search Header */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center space-x-4 mb-4">
          {/* Search Input */}
          <div className="flex-1 relative">
            <div className="relative">
              <SearchIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                ref={searchInputRef}
                type="text"
                placeholder="ค้นหาสินค้าด้วย AI Hybrid Search..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyPress={handleKeyPress}
                onFocus={() => setShowSuggestions(true)}
                onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
                className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-lg"
              />
            </div>

            {/* Search Suggestions */}
            <AnimatePresence>
              {showSuggestions && (query || searchHistory.length > 0) && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="absolute top-full left-0 right-0 mt-1 bg-white border border-gray-200 rounded-lg shadow-lg z-10"
                >
                  {query && suggestions.length > 0 && (
                    <div className="p-2">
                      <div className="text-xs text-gray-500 mb-2">คำแนะนำ</div>
                      {suggestions.map((suggestion, index) => (
                        <button
                          key={index}
                          onClick={() => {
                            setQuery(suggestion)
                            setShowSuggestions(false)
                            searchInputRef.current?.focus()
                          }}
                          className="w-full text-left px-3 py-2 hover:bg-gray-50 rounded text-sm"
                        >
                          <SearchIcon className="w-4 h-4 inline mr-2 text-gray-400" />
                          {suggestion}
                        </button>
                      ))}
                    </div>
                  )}

                  {searchHistory.length > 0 && (
                    <div className="p-2 border-t border-gray-100">
                      <div className="text-xs text-gray-500 mb-2">ค้นหาล่าสุด</div>
                      {searchHistory.slice(0, 3).map((historyItem, index) => (
                        <button
                          key={index}
                          onClick={() => {
                            setQuery(historyItem)
                            setShowSuggestions(false)
                            searchInputRef.current?.focus()
                          }}
                          className="w-full text-left px-3 py-2 hover:bg-gray-50 rounded text-sm"
                        >
                          <ClockIcon className="w-4 h-4 inline mr-2 text-gray-400" />
                          {historyItem}
                        </button>
                      ))}
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Search Button */}
          <button
            onClick={handleSearch}
            disabled={!query.trim() || isSearching}
            className="btn-premium px-6 py-3 disabled:opacity-50"
          >
            {isSearching ? (
              <>
                <div className="animate-spin w-5 h-5 border-2 border-white border-t-transparent rounded-full mr-2" />
                ค้นหา...
              </>
            ) : (
              <>
                <SearchIcon className="w-5 h-5 mr-2" />
                ค้นหา
              </>
            )}
          </button>

          {/* Filters Toggle */}
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`btn-secondary ${showFilters ? 'bg-blue-50 text-blue-600' : ''}`}
          >
            <FilterIcon className="w-5 h-5 mr-2" />
            ตัวกรอง
          </button>
        </div>

        {/* Search Stats */}
        {results.length > 0 && (
          <div className="flex items-center space-x-6 text-sm text-gray-600">
            <span>พบ {searchStats.totalResults} รายการ</span>
            <span>ใช้เวลา {searchStats.searchTime}ms</span>
            <div className="flex items-center space-x-4">
              <span className="flex items-center">
                <ZapIcon className="w-4 h-4 text-purple-500 mr-1" />
                Semantic: {searchStats.vectorMatches}
              </span>
              <span className="flex items-center">
                <TagIcon className="w-4 h-4 text-blue-500 mr-1" />
                Keyword: {searchStats.textMatches}
              </span>
              <span className="flex items-center">
                <TargetIcon className="w-4 h-4 text-green-500 mr-1" />
                Hybrid: {searchStats.hybridMatches}
              </span>
            </div>
          </div>
        )}

        {/* Advanced Filters */}
        <AnimatePresence>
          {showFilters && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-4 p-4 bg-gray-50 rounded-lg"
            >
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Categories */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    หมวดหมู่
                  </label>
                  <select
                    multiple
                    value={filters.categories}
                    onChange={(e) => {
                      const values = Array.from(e.target.selectedOptions, option => option.value)
                      setFilters(prev => ({ ...prev, categories: values }))
                    }}
                    className="input-premium h-20"
                  >
                    {categories.map(cat => (
                      <option key={cat.id} value={cat.id}>{cat.name_th}</option>
                    ))}
                  </select>
                </div>

                {/* Match Type */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    ประเภทการจับคู่
                  </label>
                  <select
                    value={filters.matchType}
                    onChange={(e) => setFilters(prev => ({ ...prev, matchType: e.target.value as any }))}
                    className="input-premium"
                  >
                    <option value="all">ทั้งหมด</option>
                    <option value="vector">Semantic Search</option>
                    <option value="text">Keyword Search</option>
                    <option value="hybrid">Hybrid Search</option>
                  </select>
                </div>

                {/* Sort */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    เรียงลำดับ
                  </label>
                  <div className="flex space-x-2">
                    <select
                      value={filters.sortBy}
                      onChange={(e) => setFilters(prev => ({ ...prev, sortBy: e.target.value as any }))}
                      className="input-premium flex-1"
                    >
                      <option value="relevance">ความเกี่ยวข้อง</option>
                      <option value="price">ราคา</option>
                      <option value="confidence">ความเชื่อมั่น</option>
                      <option value="date">วันที่</option>
                    </select>
                    <button
                      onClick={() => setFilters(prev => ({ 
                        ...prev, 
                        sortOrder: prev.sortOrder === 'asc' ? 'desc' : 'asc' 
                      }))}
                      className="btn-secondary px-3"
                    >
                      {filters.sortOrder === 'asc' ? 
                        <SortAscIcon className="w-4 h-4" /> : 
                        <SortDescIcon className="w-4 h-4" />
                      }
                    </button>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Search Results */}
      <div className="p-6">
        {isSearching ? (
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-4" />
              <p className="text-gray-600">กำลังค้นหาด้วย AI...</p>
            </div>
          </div>
        ) : results.length > 0 ? (
          <div className="space-y-4">
            {results.map((result, index) => (
              <motion.div
                key={`${result.product.id}-${index}`}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="p-4 border border-gray-200 rounded-lg hover:shadow-md transition-shadow"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                      {getMatchTypeIcon(result.matchType)}
                      <span className="text-xs text-gray-500 uppercase tracking-wide">
                        {getMatchTypeLabel(result.matchType)}
                      </span>
                      <div className="flex items-center">
                        <TrendingUpIcon className="w-4 h-4 text-green-500 mr-1" />
                        <span className="text-sm font-medium text-green-600">
                          {(result.score * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>

                    <h3 className="text-lg font-medium text-gray-900 mb-2">
                      {highlightTokens(result.product.name_th, result.matchedTokens)}
                    </h3>

                    {result.product.description && (
                      <p className="text-gray-600 mb-2">
                        {highlightTokens(result.product.description, result.matchedTokens)}
                      </p>
                    )}

                    <div className="flex items-center space-x-4 text-sm text-gray-500">
                      {result.product.category && (
                        <span>หมวดหมู่: {result.product.category.name_th}</span>
                      )}
                      {result.product.price && (
                        <span>ราคา: {result.product.price.toLocaleString()} บาท</span>
                      )}
                    </div>

                    {result.matchedTokens.length > 0 && (
                      <div className="mt-2">
                        <span className="text-xs text-gray-500 mr-2">Matched tokens:</span>
                        <div className="inline-flex flex-wrap gap-1">
                          {result.matchedTokens.map((token, i) => (
                            <span key={i} className="text-xs bg-yellow-100 text-yellow-800 px-2 py-1 rounded">
                              {token}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {result.explanation && (
                      <div className="mt-2 text-xs text-gray-600 italic">
                        {result.explanation}
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        ) : query && !isSearching ? (
          <div className="text-center py-12">
            <SearchIcon className="w-12 h-12 text-gray-300 mx-auto mb-4" />
            <p className="text-gray-600">ไม่พบผลลัพธ์สำหรับ "{query}"</p>
            <p className="text-sm text-gray-500 mt-2">ลองใช้คำค้นหาอื่น หรือปรับตัวกรอง</p>
          </div>
        ) : (
          <div className="text-center py-12">
            <SearchIcon className="w-12 h-12 text-gray-300 mx-auto mb-4" />
            <p className="text-gray-600">เริ่มค้นหาสินค้าด้วย AI Hybrid Search</p>
            <p className="text-sm text-gray-500 mt-2">รองรับการค้นหาด้วย Semantic และ Keyword</p>
          </div>
        )}
      </div>
    </div>
  )
}
