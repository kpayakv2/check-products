'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { toast } from 'react-hot-toast'
import Sidebar from '@/components/Layout/Sidebar'
import Header from '@/components/Layout/Header'
import Modal, { FOCUS_RING } from '@/components/ui/Modal'
import { 
  DatabaseService, 
  PENDING_REVIEW_STATUSES,
  Product, 
  ProductStatus,
  TaxonomyNode
} from '@/utils/supabase'
import { 
  CheckCircleIcon, 
  XCircleIcon, 
  ClockIcon, 
  SearchIcon,
  FilterIcon,
  EyeIcon,
  AlertCircleIcon,
  ShoppingBagIcon,
  TagIcon,
  DollarSignIcon,
  CalendarIcon,
  UserIcon,
  ArrowRightIcon,
  PackageIcon,
  BarChartIcon,
  InfoIcon,
  XIcon,
  ActivityIcon,
  TrendingUpIcon,
  SparklesIcon,
  Settings2Icon,
  ChevronRightIcon
} from 'lucide-react'

interface ProductFilters {
  status: string
  category: string
  search: string
}

const PAGE_SIZE = 50

export default function ProductsPage() {
  const [products, setProducts] = useState<Product[]>([])
  const [categories, setCategories] = useState<TaxonomyNode[]>([])
  const [total, setTotal] = useState(0)
  const [statusCounts, setStatusCounts] = useState<Record<string, number>>({})
  const [offset, setOffset] = useState(0)
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [showProductDetail, setShowProductDetail] = useState(false)

  const [filters, setFilters] = useState<ProductFilters>({
    status: '',
    category: '',
    search: ''
  })

  useEffect(() => {
    let cancelled = false

    const load = async () => {
      try {
        setIsLoading(true)
        const result = await DatabaseService.searchProducts({
          status: (filters.status || undefined) as ProductStatus | undefined,
          categoryId: filters.category || undefined,
          search: filters.search || undefined,
          limit: PAGE_SIZE,
          offset
        })
        if (cancelled) return
        setProducts(result.products)
        setTotal(result.total)
      } catch (error) {
        console.error('Error loading products data:', error)
        toast.error('ไม่สามารถโหลดข้อมูลสินค้าได้')
      } finally {
        if (!cancelled) setIsLoading(false)
      }
    }

    // หน่วงไว้เล็กน้อยเพื่อไม่ยิง query ทุกตัวอักษรที่พิมพ์
    const timer = setTimeout(load, filters.search ? 300 : 0)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [filters, offset])

  useEffect(() => {
    const loadContext = async () => {
      try {
        const [taxonomyData, counts] = await Promise.all([
          DatabaseService.getTaxonomyTree(),
          DatabaseService.getProductStatusCounts()
        ])
        setCategories(taxonomyData || [])
        setStatusCounts(counts)
      } catch (error) {
        console.error('Error loading products context:', error)
      }
    }
    loadContext()
  }, [])

  /** เปลี่ยนตัวกรองแล้วต้องกลับไปหน้าแรกเสมอ ไม่งั้นค้างอยู่หน้าที่ผลลัพธ์ชุดใหม่ไปไม่ถึง */
  const updateFilters = (next: Partial<ProductFilters>) => {
    setFilters(current => ({ ...current, ...next }))
    setOffset(0)
  }

  const pendingCount =
    PENDING_REVIEW_STATUSES.reduce((sum, status) => sum + (statusCounts[status] || 0), 0)

  const handleProductSelect = (product: Product) => {
    setSelectedProduct(product)
    setShowProductDetail(true)
  }

  const getStatusBadge = (status: Product['status']) => {
    switch (status) {
      case 'pending_review_dedup':
        return (
          <span className="inline-flex items-center px-3 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider bg-amber-100 text-amber-900 border border-amber-300 shadow-sm thai-text">
            <ClockIcon className="mr-1.5 h-3.5 w-3.5" aria-hidden="true" />
            รอตรวจของซ้ำ
          </span>
        )
      case 'pending_review_category':
        return (
          <span className="inline-flex items-center px-3 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider bg-sky-100 text-sky-900 border border-sky-300 shadow-sm thai-text">
            <ClockIcon className="mr-1.5 h-3.5 w-3.5" aria-hidden="true" />
            รอตรวจหมวดหมู่
          </span>
        )
      case 'approved':
        return (
          <span className="inline-flex items-center px-3 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider bg-emerald-100 text-emerald-900 border border-emerald-300 shadow-sm">
            <CheckCircleIcon className="mr-1.5 h-4 w-4" aria-hidden="true" />
            Verified Success
          </span>
        )
      case 'rejected':
        return (
          <span className="inline-flex items-center px-3 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider bg-rose-100 text-rose-900 border border-rose-300 shadow-sm">
            <XCircleIcon className="mr-1.5 h-4 w-4" aria-hidden="true" />
            Rejected Duplicate
          </span>
        )
      default:
        return <span className="text-slate-600 text-xs font-medium">ไม่ทราบสถานะ</span>
    }
  }

  if (isLoading) {
    return (
      <div className="flex h-screen bg-white font-sans">
        <Sidebar />
        <div className="flex-1 flex flex-col items-center justify-center relative overflow-hidden">
           <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-indigo-50/50 rounded-full blur-[100px]" />
           <div className="relative flex flex-col items-center">
              <div className="w-20 h-20 relative flex items-center justify-center">
                 <div className="absolute inset-0 border-4 border-indigo-100 rounded-full" />
                 <div className="absolute inset-0 border-4 border-indigo-600 rounded-full border-t-transparent animate-spin" />
                 <ShoppingBagIcon className="w-8 h-8 text-indigo-600 motion-safe:animate-pulse" />
              </div>
              <p className="mt-8 text-xs font-bold text-slate-400 uppercase tracking-widest thai-text motion-safe:animate-pulse">Syncing Inventory...</p>
           </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-screen bg-gray-50 font-sans">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden relative">
        <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-indigo-50/30 rounded-full blur-[120px] -mr-64 -mt-64 pointer-events-none" />
        <Header title="คลังสินค้า" subtitle="Product Stock Browser" />
        
        <main className="flex-1 overflow-x-hidden overflow-y-auto p-10 relative z-10">
          <div className="max-w-7xl mx-auto">
            {/* Header Area */}
            <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between mb-16 gap-8">
               <div>
                  <div className="flex items-center gap-3 mb-4">
                     <span className="px-4 py-1 bg-white border border-slate-100 rounded-full text-xs font-bold text-slate-400 uppercase tracking-wider shadow-sm">Product Registry</span>
                     <span className="px-4 py-1 bg-indigo-600 rounded-full text-xs font-bold text-white uppercase tracking-wider shadow-lg shadow-indigo-100">Read Only</span>
                  </div>
                  <h1 className="text-3xl font-black text-slate-900 tracking-tight uppercase thai-text">
                    Inventory Audit
                  </h1>
                  <p className="text-slate-500 mt-4 text-lg font-medium thai-text max-w-2xl leading-relaxed">
                    ค้นและดูสินค้าทั้งหมดในสตอก — หน้านี้ดูอย่างเดียว งานตรวจอยู่ที่ศูนย์ตรวจสอบข้อมูล
                  </p>
               </div>

               <div className="flex items-center gap-8 bg-white/60 backdrop-blur-md p-6 rounded-[32px] border border-slate-200 shadow-md">
                  <div className="flex flex-col items-center">
                     <span className="text-xs font-bold text-slate-600 uppercase tracking-wider mb-1">รอตรวจ</span>
                     <span data-testid="pending-count" className="text-xl font-black text-amber-600">{pendingCount.toLocaleString()}</span>
                  </div>
                  <div className="w-[1px] h-10 bg-slate-200" />
                  <div className="flex flex-col items-center uppercase text-xs font-bold text-slate-600 tracking-wider">
                     <span className="mb-1">ตรวจแล้ว</span>
                     <span className="text-xl font-black text-emerald-600 tracking-tight">{(statusCounts.approved || 0).toLocaleString()}</span>
                  </div>
               </div>
            </div>

            {/* Filter Portal */}
            <div className="p-8 mb-10 bg-white rounded-3xl border border-slate-200 shadow-sm relative overflow-hidden">
               <div className="flex flex-col lg:flex-row gap-6 items-center">
                  <div className="relative group flex-1 w-full">
                    <label htmlFor="product-search-input" className="sr-only">
                      ค้นหาสินค้า (ชื่อสินค้า, แบรนด์, หรือ SKU)
                    </label>
                    <SearchIcon className="absolute left-6 top-1/2 transform -translate-y-1/2 text-slate-400 h-5 w-5 group-focus-within:text-indigo-600 transition-colors pointer-events-none" aria-hidden="true" />
                    <input
                      id="product-search-input"
                      type="text"
                      placeholder="ค้นหา SKU, แบรนด์, หรือชื่อสินค้า..."
                      data-testid="product-search"
                      value={filters.search}
                      onChange={(e) => updateFilters({ search: e.target.value })}
                      className={`w-full pl-16 pr-8 py-4 bg-slate-50 border border-slate-200 rounded-2xl ${FOCUS_RING} font-medium text-slate-800 placeholder:text-slate-400 text-sm`}
                    />
                  </div>

                  <div className="flex gap-4 w-full lg:w-auto">
                    <div className="relative group flex-1 lg:w-64">
                       <label htmlFor="product-status-select" className="sr-only">
                         กรองตามสถานะสินค้า
                       </label>
                       <FilterIcon className="absolute left-6 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400 group-focus-within:text-indigo-600 pointer-events-none" aria-hidden="true" />
                       <select
                         id="product-status-select"
                         value={filters.status}
                         onChange={(e) => updateFilters({ status: e.target.value })}
                         className={`w-full pl-14 pr-10 py-4 bg-white border border-slate-200 rounded-2xl appearance-none font-bold text-xs text-slate-700 tracking-wide ${FOCUS_RING} cursor-pointer`}
                        >
                          <option value="">สถานะทั้งหมด (All Statuses)</option>
                          <option value="pending_review_dedup">🟡 รอตรวจของซ้ำ</option>
                          <option value="pending_review_category">🔵 รอตรวจหมวดหมู่</option>
                          <option value="approved">🟢 ตรวจสอบแล้ว (Verified)</option>
                          <option value="rejected">🔴 ปฏิเสธ / ซ้ำ (Rejected)</option>
                        </select>
                    </div>
                  </div>
               </div>
            </div>

            {/* Inventory Table Container */}
            <div className="premium-card bg-white/40 border-white shadow-2xl relative overflow-hidden">
              <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-50/20 rounded-full blur-[80px]" />
              
              {products.length === 0 ? (
                <div className="py-32 text-center flex flex-col items-center">
                   <div className="p-10 bg-slate-50 rounded-full mb-8 border border-slate-100">
                      <SearchIcon className="w-16 h-16 text-slate-200" />
                   </div>
                   <h3 className="text-2xl font-black text-slate-900 uppercase tracking-tight thai-text">No Results Found</h3>
                   <p className="text-slate-400 uppercase text-xs font-black tracking-widest mt-2">Adjust your audit filters</p>
                </div>
              ) : (
                <div className="overflow-x-auto custom-scrollbar">
                  <table className="w-full text-left border-collapse">
                    <caption className="sr-only">รายการสินค้าในสตอกและสถานะการตรวจสอบ</caption>
                    <thead>
                      <tr className="border-b border-slate-200 bg-slate-50/50">
                        <th scope="col" className="px-8 py-5 text-xs font-bold text-slate-700 uppercase tracking-wider thai-text">ข้อมูลสินค้า</th>
                        <th scope="col" className="px-8 py-5 text-xs font-bold text-slate-700 uppercase tracking-wider thai-text">หมวดหมู่</th>
                        <th scope="col" className="px-8 py-5 text-xs font-bold text-slate-700 uppercase tracking-wider thai-text">ราคา</th>
                        <th scope="col" className="px-8 py-5 text-xs font-bold text-slate-700 uppercase tracking-wider thai-text">สถานะ</th>
                        <th scope="col" className="px-8 py-5 text-right text-xs font-bold text-slate-700 uppercase tracking-wider">จัดการ</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {products.map((product, idx) => (
                        <motion.tr
                          key={product.id}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: idx * 0.02 }}
                          onClick={() => handleProductSelect(product)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' || e.key === ' ') {
                              e.preventDefault()
                              handleProductSelect(product)
                            }
                          }}
                          tabIndex={0}
                          role="button"
                          aria-label={`ดูรายละเอียด ${product.name_th}`}
                          className={`group hover:bg-indigo-50/40 transition-colors cursor-pointer ${FOCUS_RING}`}
                        >
                          <td className="px-8 py-6">
                             <div className="flex items-center gap-5">
                                <div className="w-14 h-14 bg-white rounded-2xl border border-slate-200 shadow-sm flex items-center justify-center transition-all">
                                   <ShoppingBagIcon className="w-6 h-6 text-indigo-500" aria-hidden="true" />
                                </div>
                                <div className="flex flex-col">
                                   <span className="text-base font-bold text-slate-900 thai-text tracking-tight group-hover:text-indigo-600 transition-colors">{product.name_th}</span>
                                   <div className="flex items-center gap-2 mt-1.5">
                                      <span className="text-xs font-medium text-slate-600 uppercase">{product.brand || 'No Brand'}</span>
                                      <span className="w-1 h-1 bg-slate-300 rounded-full" />
                                      <span className="text-xs font-medium text-indigo-600">{product.sku || 'No SKU'}</span>
                                   </div>
                                </div>
                             </div>
                          </td>
                          <td className="px-8 py-6">
                             <div className="flex flex-col">
                                <span className="text-sm font-semibold text-slate-800 thai-text">{product.category?.name_th || 'ไม่ได้ระบุหมวดหมู่'}</span>
                                <span className="text-xs text-slate-500 mt-0.5">หมวดหมู่ระบบ</span>
                             </div>
                          </td>
                          <td className="px-8 py-6">
                             <div className="flex flex-col">
                                <span className="text-lg font-bold text-slate-900">฿{product.price?.toLocaleString() || '0.00'}</span>
                                <span className="text-xs text-slate-500 mt-0.5">ราคาลงทะเบียน</span>
                             </div>
                          </td>
                          <td className="px-8 py-6">
                             {getStatusBadge(product.status)}
                          </td>
                          <td className="px-8 py-6 text-right">
                             <button
                               type="button"
                               aria-label={`ดูรายละเอียด ${product.name_th}`}
                               onClick={(e) => {
                                 e.stopPropagation()
                                 handleProductSelect(product)
                               }}
                               className={`w-10 h-10 bg-white border border-slate-200 text-slate-500 rounded-xl inline-flex items-center justify-center hover:bg-slate-50 hover:text-indigo-600 hover:border-indigo-300 hover:shadow-sm transition-all ${FOCUS_RING}`}
                             >
                                <ChevronRightIcon className="w-4 h-4" aria-hidden="true" />
                             </button>
                          </td>
                        </motion.tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {/* แบ่งหน้า — สตอกมีหลายพันแถว หน้านี้จึงโหลดทีละหน้าจากฐานข้อมูล */}
            <div className="flex items-center justify-between gap-6 mt-10 px-4">
               <p data-testid="result-range" className="text-xs font-semibold text-slate-600 tracking-wide">
                 {total === 0
                   ? 'ไม่พบรายการ'
                   : `${(offset + 1).toLocaleString()}–${Math.min(offset + PAGE_SIZE, total).toLocaleString()} จาก ${total.toLocaleString()}`}
               </p>
               <div className="flex gap-3">
                  <button
                    data-testid="prev-page"
                    type="button"
                    onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
                    disabled={offset === 0}
                    className={`px-8 py-3.5 bg-white border border-slate-200 rounded-full text-xs font-bold text-slate-700 uppercase tracking-wider disabled:opacity-40 disabled:cursor-not-allowed hover:bg-slate-50 hover:shadow-sm transition-all ${FOCUS_RING}`}
                  >
                    ก่อนหน้า
                  </button>
                  <button
                    data-testid="next-page"
                    type="button"
                    onClick={() => setOffset(offset + PAGE_SIZE)}
                    disabled={offset + PAGE_SIZE >= total}
                    className={`px-8 py-3.5 bg-slate-900 text-white rounded-full text-xs font-bold uppercase tracking-wider disabled:opacity-40 disabled:cursor-not-allowed hover:bg-indigo-600 transition-all ${FOCUS_RING}`}
                  >
                    ถัดไป
                  </button>
               </div>
            </div>
          </div>
        </main>
      </div>

      {/* Product Detail Modal */}
      <Modal
        isOpen={showProductDetail && !!selectedProduct}
        onClose={() => setShowProductDetail(false)}
        maxWidth="max-w-5xl"
        dataTestId="product-detail"
        showCloseButton={false}
        ariaLabel={selectedProduct ? `รายละเอียดสินค้า ${selectedProduct.name_th}` : 'รายละเอียดสินค้า'}
        className="rounded-[40px] p-0"
      >
        {selectedProduct && (
          <div className="-m-6">
            {/* Hero Gradient Header */}
            <div className="h-48 bg-indigo-600 relative overflow-hidden p-8 sm:p-12 flex justify-between items-start">
               <div className="absolute top-0 right-0 w-96 h-96 bg-white/10 rounded-full blur-[80px] -mr-32 -mt-32 pointer-events-none" />
               <div className="absolute bottom-0 left-0 w-64 h-64 bg-indigo-400/20 rounded-full blur-[60px] -ml-32 -mb-32 pointer-events-none" />
               
               <div className="relative z-10 flex items-center gap-6 sm:gap-10">
                  <div className="w-20 h-20 sm:w-24 sm:h-24 bg-white/20 backdrop-blur-md rounded-[28px] border border-white/30 flex items-center justify-center text-white shadow-2xl shrink-0">
                     <SparklesIcon className="w-8 h-8 sm:w-10 sm:h-10" aria-hidden="true" />
                  </div>
                  <div>
                     <div className="flex items-center gap-3 mb-2">
                        <span className="px-3 py-1 bg-white/20 backdrop-blur-md rounded-full text-xs font-bold text-white uppercase tracking-wider border border-white/20">Audit Inspect</span>
                        <div className="w-2 h-2 bg-emerald-400 rounded-full motion-safe:animate-ping" aria-hidden="true" />
                     </div>
                     <h2 className="text-2xl sm:text-3xl font-black text-white thai-text tracking-tight uppercase leading-tight">{selectedProduct.name_th}</h2>
                  </div>
               </div>

               <button
                 type="button"
                 onClick={() => setShowProductDetail(false)}
                 aria-label="ปิดหน้าต่างรายละเอียดสินค้า"
                 className={`relative z-10 w-12 h-12 bg-white/10 hover:bg-white/20 rounded-full flex items-center justify-center text-white transition-all backdrop-blur-md border border-white/20 ${FOCUS_RING}`}
               >
                  <XIcon className="w-6 h-6" aria-hidden="true" />
               </button>
            </div>

            {/* Content Matrix */}
            <div className="p-8 sm:p-12 grid grid-cols-1 lg:grid-cols-3 gap-8">
               <div className="lg:col-span-2 space-y-8">
                  <section>
                     <h3 className="text-xs font-bold text-indigo-700 uppercase tracking-wider mb-4 flex items-center gap-3">
                        <div className="w-1.5 h-4 bg-indigo-600 rounded-full" aria-hidden="true" />
                        ข้อมูลทางเทคนิค (Technical Specifications)
                     </h3>
                     <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        {[
                           { label: 'Merchant SKU', value: selectedProduct.sku || 'ไม่ระบุ' },
                           { label: 'Global Brand', value: selectedProduct.brand || 'ไม่ระบุแบรนด์' },
                           { label: 'Model Tier', value: selectedProduct.model || 'มาตรฐาน' },
                           { label: 'Taxonomy Category', value: selectedProduct.category?.name_th || 'รอการจำแนก' }
                        ].map((item, i) => (
                           <div key={i} className="p-6 bg-slate-50 rounded-2xl border border-slate-200">
                              <p className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-1">{item.label}</p>
                              <p className="text-base font-bold text-slate-800 thai-text tracking-tight">{item.value}</p>
                           </div>
                        ))}
                     </div>
                  </section>

                  <section>
                     <h3 className="text-xs font-bold text-indigo-700 uppercase tracking-wider mb-4">
                        รายละเอียดเพิ่มเติม (Description)
                     </h3>
                     <div className="p-6 bg-slate-900 rounded-2xl text-slate-200 text-base font-medium leading-relaxed thai-text relative overflow-hidden">
                        <ActivityIcon className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-48 text-white/5 pointer-events-none" aria-hidden="true" />
                        &ldquo;{selectedProduct.description || 'ไม่มีรายละเอียดเพิ่มเติมสำหรับรายการนี้'}&rdquo;
                     </div>
                  </section>
               </div>

               <div className="space-y-8">
                  <section className="bg-indigo-50/70 p-8 rounded-2xl border border-indigo-200 flex flex-col justify-between">
                     <div>
                        <p className="text-xs font-bold text-indigo-800 uppercase tracking-wider mb-4">ราคาลงทะเบียน (Registry Value)</p>
                        <div className="flex items-baseline gap-2">
                           <span className="text-4xl font-black text-indigo-700 tracking-tight">฿{selectedProduct.price?.toLocaleString()}</span>
                           <span className="text-xs font-bold text-indigo-500 uppercase">THB</span>
                        </div>
                     </div>
                  </section>

                  {PENDING_REVIEW_STATUSES.includes(selectedProduct.status) && (
                    <section className="p-8 bg-amber-50 rounded-2xl border border-amber-200 space-y-4">
                       <p className="text-xs font-bold text-amber-800 uppercase tracking-wider">รอการตรวจ</p>
                       <p className="text-sm font-medium text-slate-700 thai-text leading-relaxed">
                         สินค้าชิ้นนี้ยังรอคนตรวจอยู่ หน้านี้ดูอย่างเดียว — งานตรวจทั้งหมดอยู่ที่ศูนย์ตรวจสอบข้อมูล
                       </p>
                       <a
                         href="/data-quality"
                         className={`inline-flex items-center gap-2 px-6 py-3 bg-slate-900 text-white rounded-xl text-xs font-bold uppercase tracking-wider hover:bg-indigo-600 transition-all ${FOCUS_RING}`}
                       >
                         ไปที่ศูนย์ตรวจสอบข้อมูล
                         <ArrowRightIcon className="w-4 h-4" aria-hidden="true" />
                       </a>
                    </section>
                  )}
               </div>
            </div>
          </div>
        )}
      </Modal>
    </div>
  )
}
