'use client'

import { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { toast } from 'react-hot-toast'
import { 
  CheckCircleIcon,
  XCircleIcon,
  EditIcon,
  SearchIcon,
  FilterIcon,
  KeyboardIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  TagIcon,
  ClockIcon,
  UserIcon
} from 'lucide-react'
import { Product, TaxonomyNode, SimilarityMatch } from '@/utils/supabase'

interface EnhancedProductReviewProps {
  products: Product[]
  categories: TaxonomyNode[]
  onProductApprove?: (productId: string, categoryId?: string) => void
  onProductReject?: (productId: string, reason?: string) => void
  onProductUpdate?: (productId: string, updates: Partial<Product>) => void
  onSimilarityCheck?: (productId: string) => Promise<SimilarityMatch[]>
  className?: string
}

interface ProductFilters {
  status: 'all' | 'pending' | 'approved' | 'rejected'
  category: string
  confidence: 'all' | 'high' | 'medium' | 'low'
  search: string
}

export default function EnhancedProductReview({
  products,
  categories,
  onProductApprove,
  onProductReject,
  onProductUpdate,
  onSimilarityCheck,
  className = ''
}: EnhancedProductReviewProps) {
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null)
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [showSidePanel, setShowSidePanel] = useState(true)
  const [similarityMatches, setSimilarityMatches] = useState<SimilarityMatch[]>([])
  const [showKeyboardHelp, setShowKeyboardHelp] = useState(false)
  const [isLoading, setIsLoading] = useState(false)

  const [filters, setFilters] = useState<ProductFilters>({
    status: 'pending',
    category: '',
    confidence: 'all',
    search: ''
  })

  // Filter products
  const filteredProducts = products.filter(product => {
    const matchesStatus = filters.status === 'all' || product.status === filters.status
    const matchesCategory = !filters.category || product.category_id === filters.category
    const matchesSearch = !filters.search || 
      product.name_th.toLowerCase().includes(filters.search.toLowerCase()) ||
      product.description?.toLowerCase().includes(filters.search.toLowerCase())
    
    let matchesConfidence = true
    if (filters.confidence !== 'all' && product.confidence_score) {
      const score = product.confidence_score
      matchesConfidence = 
        (filters.confidence === 'high' && score >= 0.8) ||
        (filters.confidence === 'medium' && score >= 0.5 && score < 0.8) ||
        (filters.confidence === 'low' && score < 0.5)
    }

    return matchesStatus && matchesCategory && matchesSearch && matchesConfidence
  })

  // Keyboard shortcuts
  const handleKeyPress = useCallback((event: KeyboardEvent) => {
    if (!selectedProduct) return

    switch (event.key) {
      case 'a':
      case 'A':
        if (event.ctrlKey || event.metaKey) return
        event.preventDefault()
        handleApprove()
        break
      case 'r':
      case 'R':
        if (event.ctrlKey || event.metaKey) return
        event.preventDefault()
        handleReject()
        break
      case 'ArrowUp':
        event.preventDefault()
        navigateProduct(-1)
        break
      case 'ArrowDown':
        event.preventDefault()
        navigateProduct(1)
        break
      case 'Escape':
        setSelectedProduct(null)
        break
      case '?':
        setShowKeyboardHelp(true)
        break
    }
  }, [selectedProduct, selectedIndex])

  useEffect(() => {
    document.addEventListener('keydown', handleKeyPress)
    return () => document.removeEventListener('keydown', handleKeyPress)
  }, [handleKeyPress])

  // Load similarity matches when product is selected
  useEffect(() => {
    if (selectedProduct && onSimilarityCheck) {
      setIsLoading(true)
      onSimilarityCheck(selectedProduct.id)
        .then(setSimilarityMatches)
        .catch(() => setSimilarityMatches([]))
        .finally(() => setIsLoading(false))
    }
  }, [selectedProduct, onSimilarityCheck])

  const navigateProduct = (direction: number) => {
    const newIndex = Math.max(0, Math.min(filteredProducts.length - 1, selectedIndex + direction))
    setSelectedIndex(newIndex)
    setSelectedProduct(filteredProducts[newIndex])
  }

  const handleApprove = () => {
    if (!selectedProduct) return
    onProductApprove?.(selectedProduct.id, selectedProduct.category_id)
    toast.success('อนุมัติสินค้าแล้ว')
    navigateProduct(1)
  }

  const handleReject = () => {
    if (!selectedProduct) return
    onProductReject?.(selectedProduct.id, 'Rejected via review')
    toast.error('ปฏิเสธสินค้าแล้ว')
    navigateProduct(1)
  }

  const getConfidenceColor = (score?: number) => {
    if (!score) return 'text-gray-500'
    if (score >= 0.8) return 'text-green-600'
    if (score >= 0.5) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getConfidenceBadge = (score?: number) => {
    if (!score) return 'ไม่ระบุ'
    if (score >= 0.8) return 'สูง'
    if (score >= 0.5) return 'ปานกลาง'
    return 'ต่ำ'
  }

  return (
    <div className={`bg-white rounded-lg border border-gray-200 ${className}`}>
      {/* Header */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900">ตรวจสอบสินค้า</h2>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowKeyboardHelp(true)}
              className="btn-secondary"
            >
              <KeyboardIcon className="w-4 h-4 mr-2" />
              Shortcuts
            </button>
            
            <button
              onClick={() => setShowSidePanel(!showSidePanel)}
              className="btn-secondary"
            >
              {showSidePanel ? 'ซ่อน Panel' : 'แสดง Panel'}
            </button>
          </div>
        </div>

        {/* Filters */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          <div className="relative">
            <SearchIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="ค้นหาสินค้า..."
              value={filters.search}
              onChange={(e) => setFilters(prev => ({ ...prev, search: e.target.value }))}
              className="input-premium pl-10"
            />
          </div>

          <select
            value={filters.status}
            onChange={(e) => setFilters(prev => ({ ...prev, status: e.target.value as any }))}
            className="input-premium"
          >
            <option value="all">ทุกสถานะ</option>
            <option value="pending">รอตรวจสอบ</option>
            <option value="approved">อนุมัติแล้ว</option>
            <option value="rejected">ปฏิเสธแล้ว</option>
          </select>

          <select
            value={filters.category}
            onChange={(e) => setFilters(prev => ({ ...prev, category: e.target.value }))}
            className="input-premium"
          >
            <option value="">ทุกหมวดหมู่</option>
            {categories.map(cat => (
              <option key={cat.id} value={cat.id}>{cat.name_th}</option>
            ))}
          </select>

          <select
            value={filters.confidence}
            onChange={(e) => setFilters(prev => ({ ...prev, confidence: e.target.value as any }))}
            className="input-premium"
          >
            <option value="all">ทุกระดับความเชื่อมั่น</option>
            <option value="high">สูง (≥80%)</option>
            <option value="medium">ปานกลาง (50-79%)</option>
            <option value="low">ต่ำ (&lt;50%)</option>
          </select>

          <div className="flex items-center text-sm text-gray-600">
            <FilterIcon className="w-4 h-4 mr-1" />
            {filteredProducts.length} รายการ
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex h-96">
        {/* Product Table */}
        <div className={`${showSidePanel ? 'w-1/2' : 'w-full'} border-r border-gray-200`}>
          <div className="overflow-y-auto h-full">
            <table className="w-full">
              <thead className="bg-gray-50 sticky top-0">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">สินค้า</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">หมวดหมู่</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">ความเชื่อมั่น</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">สถานะ</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">การดำเนินการ</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {filteredProducts.map((product, index) => (
                  <motion.tr
                    key={product.id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className={`
                      cursor-pointer transition-colors
                      ${selectedProduct?.id === product.id ? 'bg-blue-50' : 'hover:bg-gray-50'}
                    `}
                    onClick={() => {
                      setSelectedProduct(product)
                      setSelectedIndex(index)
                    }}
                  >
                    <td className="px-4 py-3">
                      <div>
                        <div className="font-medium text-gray-900 truncate max-w-xs">
                          {product.name_th}
                        </div>
                        {product.keywords && product.keywords.length > 0 && (
                          <div className="flex flex-wrap gap-1 mt-1">
                            {product.keywords.slice(0, 2).map((keyword, i) => (
                              <span key={i} className="text-xs bg-gray-100 text-gray-600 px-1 rounded">
                                {keyword}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      {product.category ? (
                        <span className="text-sm text-gray-900">{product.category.name_th}</span>
                      ) : (
                        <span className="text-sm text-gray-500">ไม่ระบุ</span>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center">
                        <span className={`text-sm font-medium ${getConfidenceColor(product.confidence_score)}`}>
                          {product.confidence_score ? `${(product.confidence_score * 100).toFixed(0)}%` : 'N/A'}
                        </span>
                        <span className={`ml-2 text-xs px-2 py-1 rounded-full ${
                          product.confidence_score && product.confidence_score >= 0.8 
                            ? 'bg-green-100 text-green-800'
                            : product.confidence_score && product.confidence_score >= 0.5
                            ? 'bg-yellow-100 text-yellow-800'
                            : 'bg-red-100 text-red-800'
                        }`}>
                          {getConfidenceBadge(product.confidence_score)}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        product.status === 'approved' 
                          ? 'bg-green-100 text-green-800'
                          : product.status === 'rejected'
                          ? 'bg-red-100 text-red-800'
                          : 'bg-yellow-100 text-yellow-800'
                      }`}>
                        {product.status === 'approved' ? 'อนุมัติ' :
                         product.status === 'rejected' ? 'ปฏิเสธ' : 'รอตรวจสอบ'}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center space-x-1">
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            onProductApprove?.(product.id, product.category_id)
                          }}
                          className="p-1 rounded hover:bg-green-100 text-green-600"
                          title="อนุมัติ (A)"
                        >
                          <CheckCircleIcon className="w-4 h-4" />
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            onProductReject?.(product.id)
                          }}
                          className="p-1 rounded hover:bg-red-100 text-red-600"
                          title="ปฏิเสธ (R)"
                        >
                          <XCircleIcon className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Side Panel */}
        {showSidePanel && selectedProduct && (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="w-1/2 flex flex-col"
          >
            {/* Product Details Header */}
            <div className="p-4 border-b border-gray-200 bg-gray-50">
              <div className="flex items-center justify-between">
                <h3 className="font-medium text-gray-900">รายละเอียดสินค้า</h3>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => navigateProduct(-1)}
                    disabled={selectedIndex === 0}
                    className="p-1 rounded hover:bg-gray-200 disabled:opacity-50"
                  >
                    <ChevronLeftIcon className="w-4 h-4" />
                  </button>
                  <span className="text-sm text-gray-600">
                    {selectedIndex + 1} / {filteredProducts.length}
                  </span>
                  <button
                    onClick={() => navigateProduct(1)}
                    disabled={selectedIndex === filteredProducts.length - 1}
                    className="p-1 rounded hover:bg-gray-200 disabled:opacity-50"
                  >
                    <ChevronRightIcon className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>

            {/* Product Info */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              <div>
                <h4 className="font-medium text-gray-900 mb-2">{selectedProduct.name_th}</h4>
                {selectedProduct.description && (
                  <p className="text-sm text-gray-600">{selectedProduct.description}</p>
                )}
              </div>

              {/* Metadata */}
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-500">หมวดหมู่:</span>
                  <div className="font-medium">{selectedProduct.category?.name_th || 'ไม่ระบุ'}</div>
                </div>
                <div>
                  <span className="text-gray-500">ความเชื่อมั่น:</span>
                  <div className={`font-medium ${getConfidenceColor(selectedProduct.confidence_score)}`}>
                    {selectedProduct.confidence_score ? `${(selectedProduct.confidence_score * 100).toFixed(0)}%` : 'N/A'}
                  </div>
                </div>
                {selectedProduct.brand && (
                  <div>
                    <span className="text-gray-500">แบรนด์:</span>
                    <div className="font-medium">{selectedProduct.brand}</div>
                  </div>
                )}
                {selectedProduct.price && (
                  <div>
                    <span className="text-gray-500">ราคา:</span>
                    <div className="font-medium">{selectedProduct.price.toLocaleString()} บาท</div>
                  </div>
                )}
              </div>

              {/* Keywords */}
              {selectedProduct.keywords && selectedProduct.keywords.length > 0 && (
                <div>
                  <h5 className="font-medium text-gray-900 mb-2">Keywords</h5>
                  <div className="flex flex-wrap gap-1">
                    {selectedProduct.keywords.map((keyword, i) => (
                      <span key={i} className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                        {keyword}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Similarity Matches */}
              <div>
                <h5 className="font-medium text-gray-900 mb-2">สินค้าที่คล้ายกัน</h5>
                {isLoading ? (
                  <div className="text-sm text-gray-500">กำลังโหลด...</div>
                ) : similarityMatches.length > 0 ? (
                  <div className="space-y-2">
                    {similarityMatches.slice(0, 3).map((match) => (
                      <div key={match.id} className="p-2 border border-gray-200 rounded text-sm">
                        <div className="font-medium">{match.product_b?.name_th}</div>
                        <div className="text-gray-600">
                          คล้ายกัน: {(match.similarity_score * 100).toFixed(1)}%
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-sm text-gray-500">ไม่พบสินค้าที่คล้ายกัน</div>
                )}
              </div>

              {/* Action Buttons */}
              <div className="flex space-x-2 pt-4 border-t border-gray-200">
                <button
                  onClick={handleApprove}
                  className="flex-1 btn-premium"
                >
                  <CheckCircleIcon className="w-4 h-4 mr-2" />
                  อนุมัติ (A)
                </button>
                <button
                  onClick={handleReject}
                  className="flex-1 bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition-colors flex items-center justify-center"
                >
                  <XCircleIcon className="w-4 h-4 mr-2" />
                  ปฏิเสธ (R)
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </div>

      {/* Keyboard Help Modal */}
      {showKeyboardHelp && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md mx-4">
            <h3 className="text-lg font-semibold mb-4">Keyboard Shortcuts</h3>
            
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>อนุมัติสินค้า</span>
                <kbd className="px-2 py-1 bg-gray-100 rounded">A</kbd>
              </div>
              <div className="flex justify-between">
                <span>ปฏิเสธสินค้า</span>
                <kbd className="px-2 py-1 bg-gray-100 rounded">R</kbd>
              </div>
              <div className="flex justify-between">
                <span>สินค้าก่อนหน้า</span>
                <kbd className="px-2 py-1 bg-gray-100 rounded">↑</kbd>
              </div>
              <div className="flex justify-between">
                <span>สินค้าถัดไป</span>
                <kbd className="px-2 py-1 bg-gray-100 rounded">↓</kbd>
              </div>
              <div className="flex justify-between">
                <span>ปิด Panel</span>
                <kbd className="px-2 py-1 bg-gray-100 rounded">Esc</kbd>
              </div>
              <div className="flex justify-between">
                <span>แสดงความช่วยเหลือ</span>
                <kbd className="px-2 py-1 bg-gray-100 rounded">?</kbd>
              </div>
            </div>
            
            <div className="flex justify-end mt-4">
              <button
                onClick={() => setShowKeyboardHelp(false)}
                className="btn-premium"
              >
                ปิด
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
