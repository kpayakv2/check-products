'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { toast } from 'react-hot-toast'
import Sidebar from '@/components/Layout/Sidebar'
import Header from '@/components/Layout/Header'
import { 
  DatabaseService, 
  Product, 
  TaxonomyNode, 
  SimilarityMatch 
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
  dateRange: string
}

export default function ProductsPage() {
  const [products, setProducts] = useState<Product[]>([])
  const [categories, setCategories] = useState<TaxonomyNode[]>([])
  const [filteredProducts, setFilteredProducts] = useState<Product[]>([])
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null)
  const [similarityMatches, setSimilarityMatches] = useState<SimilarityMatch[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [showProductDetail, setShowProductDetail] = useState(false)
  
  const [filters, setFilters] = useState<ProductFilters>({
    status: 'pending',
    category: '',
    search: '',
    dateRange: ''
  })

  useEffect(() => {
    loadData()
  }, [])

  useEffect(() => {
    let filtered = products

    if (filters.status) {
      filtered = filtered.filter(product => product.status === filters.status)
    }

    if (filters.category) {
      filtered = filtered.filter(product => product.category_id === filters.category)
    }

    if (filters.search) {
      const s = filters.search.toLowerCase()
      filtered = filtered.filter(product => 
        product.name_th.toLowerCase().includes(s) ||
        (product.name_en && product.name_en.toLowerCase().includes(s)) ||
        (product.brand && product.brand.toLowerCase().includes(s)) ||
        (product.sku && product.sku.toLowerCase().includes(s))
      )
    }

    setFilteredProducts(filtered)
  }, [products, filters])

  const loadData = async () => {
    try {
      setIsLoading(true)
      const [productData, taxonomyData] = await Promise.all([
        DatabaseService.getProducts(),
        DatabaseService.getTaxonomyTree()
      ])
      setProducts(productData || [])
      setCategories(taxonomyData || [])
    } catch (error) {
      console.error('Error loading products data:', error)
      toast.error('ไม่สามารถโหลดข้อมูลสินค้าได้')
    } finally {
      setIsLoading(false)
    }
  }

  const loadSimilarityMatches = async (productId: string) => {
    try {
      setSimilarityMatches([]) 
    } catch (error) {
      console.error('Error loading similarity matches:', error)
    }
  }

  const handleProductSelect = async (product: Product) => {
    setSelectedProduct(product)
    setShowProductDetail(true)
    await loadSimilarityMatches(product.id)
  }

  const handleProductReview = async (productId: string, status: Product['status']) => {
    try {
      await DatabaseService.updateProductStatus(productId, status, 'current-user-id')
      toast.success(`${status === 'approved' ? 'อนุมัติ' : 'ปฏิเสธ'}สินค้าเรียบร้อยแล้ว`)
      setShowProductDetail(false)
      loadData()
    } catch (error) {
      toast.error('เกิดข้อผิดพลาดในการอัปเดตสถานะ')
    }
  }

  const getStatusBadge = (status: Product['status']) => {
    switch (status) {
      case 'pending':
        return (
          <span className="inline-flex items-center px-4 py-1.5 rounded-full text-[10px] font-black uppercase tracking-widest bg-amber-50 text-amber-600 border border-amber-100/50 shadow-sm">
            <ClockIcon className="mr-2 h-3.5 w-3.5" />
            Pending Action
          </span>
        )
      case 'approved':
        return (
          <span className="inline-flex items-center px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider bg-emerald-50 text-emerald-600 border border-emerald-100/50 shadow-sm">
            <CheckCircleIcon className="mr-2 h-4 w-4" />
            Verified Success
          </span>
        )
      case 'rejected':
        return (
          <span className="inline-flex items-center px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-wider bg-rose-50 text-rose-600 border border-rose-100/50 shadow-sm">
            <XCircleIcon className="mr-2 h-4 w-4" />
            Rejected Duplicate
          </span>
        )
      default:
        return <span className="text-slate-400">Unknown</span>
    }
  }

  if (isLoading) {
    return (
      <div className="flex h-screen bg-white font-sans">
        <Sidebar />
        <div className="flex-1 flex flex-col items-center justify-center relative">
           <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-indigo-50/50 rounded-full blur-[100px]" />
           <div className="relative flex flex-col items-center">
              <div className="w-20 h-20 relative flex items-center justify-center">
                 <div className="absolute inset-0 border-4 border-indigo-100 rounded-full" />
                 <div className="absolute inset-0 border-4 border-indigo-600 rounded-full border-t-transparent animate-spin" />
                 <ShoppingBagIcon className="w-8 h-8 text-indigo-600 animate-pulse" />
              </div>
              <p className="mt-8 text-xs font-bold text-slate-400 uppercase tracking-widest thai-text animate-pulse">Syncing Inventory...</p>
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
        <Header title="คลังสินค้าและประวัติการตรวจสอบ" subtitle="Inventory Audit Dashboard" />
        
        <main className="flex-1 overflow-x-hidden overflow-y-auto p-10 relative z-10">
          <div className="max-w-7xl mx-auto">
            {/* Header Area */}
            <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between mb-16 gap-8">
               <div>
                  <div className="flex items-center gap-3 mb-4">
                     <span className="px-4 py-1 bg-white border border-slate-100 rounded-full text-xs font-bold text-slate-400 uppercase tracking-wider shadow-sm">Product Registry</span>
                     <span className="px-4 py-1 bg-indigo-600 rounded-full text-xs font-bold text-white uppercase tracking-wider shadow-lg shadow-indigo-100">Audit Active</span>
                  </div>
                  <h1 className="text-3xl font-black text-slate-900 tracking-tight uppercase thai-text">
                    Inventory Audit
                  </h1>
                  <p className="text-slate-500 mt-4 text-lg font-medium thai-text max-w-2xl leading-relaxed">
                    ตรวจสอบและยืนยันความถูกต้องของข้อมูลสินค้าก่อนการประมวลผลเข้าสู่ฐานข้อมูลหลักขององค์กร
                  </p>
               </div>

               <div className="flex items-center gap-8 bg-white/40 backdrop-blur-md p-6 rounded-[40px] border border-white shadow-xl">
                  <div className="flex flex-col items-center">
                     <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">Pending</span>
                     <span className="text-xl font-black text-amber-500">{products.filter(p=>p.status==='pending').length}</span>
                  </div>
                  <div className="w-[1px] h-10 bg-slate-100" />
                  <div className="flex flex-col items-center uppercase text-[10px] font-black text-slate-400 tracking-widest">
                     <span className="mb-1">Verified</span>
                     <span className="text-xl font-black text-emerald-500 tracking-tight">{products.filter(p=>p.status==='approved').length}</span>
                  </div>
               </div>
            </div>

            {/* Filter Portal */}
            <div className="premium-card p-10 mb-12 bg-white/60 border-white shadow-xl relative overflow-hidden">
               <div className="flex flex-col lg:flex-row gap-8 items-center">
                  <div className="relative group flex-1 w-full">
                    <SearchIcon className="absolute left-6 top-1/2 transform -translate-y-1/2 text-slate-300 h-5 w-5 group-focus-within:text-indigo-600 transition-colors" />
                    <input
                      type="text"
                      placeholder="Search SKU, Brands, or Identity Attributes..."
                      value={filters.search}
                      onChange={(e) => setFilters({ ...filters, search: e.target.value })}
                      className="w-full pl-16 pr-8 py-5 bg-slate-50/50 border border-slate-100 rounded-[32px] focus:outline-none focus:ring-4 focus:ring-indigo-500/10 focus:border-indigo-500/50 transition-all font-black text-slate-800 placeholder:text-slate-300 uppercase text-xs tracking-widest"
                    />
                  </div>

                  <div className="flex gap-4 w-full lg:w-auto">
                    <div className="relative group flex-1 lg:w-64">
                       <FilterIcon className="absolute left-6 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-300 group-focus-within:text-indigo-600" />
                       <select
                         value={filters.status}
                         onChange={(e) => setFilters({ ...filters, status: e.target.value })}
                         className="w-full pl-14 pr-10 py-5 bg-white border border-slate-100 rounded-[32px] appearance-none font-black text-[11px] text-slate-600 uppercase tracking-widest focus:ring-4 focus:ring-indigo-500/10 transition-all cursor-pointer"
                       >
                         <option value="">Status: All</option>
                         <option value="pending">🟡 Pending</option>
                         <option value="approved">🟢 Verified</option>
                         <option value="rejected">🔴 Rejected</option>
                       </select>
                    </div>
                  </div>
               </div>
            </div>

            {/* Inventory Table Container */}
            <div className="premium-card bg-white/40 border-white shadow-2xl relative overflow-hidden">
              <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-50/20 rounded-full blur-[80px]" />
              
              {filteredProducts.length === 0 ? (
                <div className="py-32 text-center flex flex-col items-center">
                   <div className="p-10 bg-slate-50 rounded-full mb-8 border border-slate-100">
                      <SearchIcon className="w-16 h-16 text-slate-200" />
                   </div>
                   <h3 className="text-2xl font-black text-slate-900 uppercase tracking-tight thai-text">No Results Found</h3>
                   <p className="text-slate-400 uppercase text-[10px] font-black tracking-widest mt-2">Adjust your audit filters</p>
                </div>
              ) : (
                <div className="overflow-x-auto custom-scrollbar">
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="border-b border-slate-100">
                        <th className="px-10 py-8 text-[11px] font-black text-slate-400 uppercase tracking-[0.3em] thai-text">Product Signature</th>
                        <th className="px-10 py-8 text-[11px] font-black text-slate-400 uppercase tracking-[0.3em] thai-text">Taxonomy Class</th>
                        <th className="px-8 py-8 text-[11px] font-black text-slate-400 uppercase tracking-[0.3em] thai-text">Valuation</th>
                        <th className="px-8 py-8 text-[11px] font-black text-slate-400 uppercase tracking-[0.3em] thai-text">Audit Trace</th>
                        <th className="px-10 py-8 text-right text-[11px] font-black text-slate-400 uppercase tracking-[0.3em]">Control</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-50/50">
                      {filteredProducts.map((product, idx) => (
                        <motion.tr
                          key={product.id}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: idx * 0.02 }}
                          onClick={() => handleProductSelect(product)}
                          className="group hover:bg-indigo-50/30 transition-all duration-300 cursor-pointer"
                        >
                          <td className="px-10 py-8">
                             <div className="flex items-center gap-6">
                                <div className="w-16 h-16 bg-white rounded-[24px] border border-slate-100 shadow-sm flex items-center justify-center group-hover:scale-110 group-hover:rotate-3 transition-all duration-500">
                                   <ShoppingBagIcon className="w-7 h-7 text-indigo-400" />
                                </div>
                                <div className="flex flex-col">
                                   <h4 className="text-lg font-black text-slate-900 thai-text tracking-tight leading-tight group-hover:text-indigo-600 transition-colors">{product.name_th}</h4>
                                   <div className="flex items-center gap-2 mt-2">
                                      <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">{product.brand || 'No Brand'}</span>
                                      <span className="w-1 h-1 bg-slate-200 rounded-full" />
                                      <span className="text-[10px] font-black text-indigo-400 tracking-tighter">{product.sku || 'No SKU'}</span>
                                   </div>
                                </div>
                             </div>
                          </td>
                          <td className="px-10 py-8">
                             <div className="flex flex-col">
                                <span className="text-sm font-black text-slate-700 thai-text tracking-tight">{product.category?.name_th || 'Unclassed'}</span>
                                <span className="text-[9px] font-black text-slate-400 uppercase tracking-[0.2em] mt-1 italic">Node Trace</span>
                             </div>
                          </td>
                          <td className="px-8 py-8">
                             <div className="flex flex-col">
                                <span className="text-xl font-black text-slate-900 tracking-tighter">฿{product.price?.toLocaleString() || '0.00'}</span>
                                <span className="text-[9px] font-black text-slate-400 uppercase tracking-widest mt-1">Registry Price</span>
                             </div>
                          </td>
                          <td className="px-8 py-8">
                             {getStatusBadge(product.status)}
                          </td>
                          <td className="px-10 py-8 text-right">
                             <button className="w-12 h-12 bg-white/50 border border-slate-100 text-slate-400 rounded-2xl flex items-center justify-center hover:bg-white hover:text-indigo-600 hover:border-indigo-200 hover:shadow-xl transition-all duration-500 overflow-hidden relative">
                                <ChevronRightIcon className="w-5 h-5" />
                             </button>
                          </td>
                        </motion.tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </main>
      </div>

      {/* Audit Detail Modal Overhaul */}
      <AnimatePresence>
        {showProductDetail && selectedProduct && (
          <div className="fixed inset-0 z-50 bg-slate-900/40 backdrop-blur-xl flex items-center justify-center p-8 overflow-y-auto">
             <motion.div
               initial={{ opacity: 0, scale: 0.9, y: 40 }}
               animate={{ opacity: 1, scale: 1, y: 0 }}
               exit={{ opacity: 0, scale: 0.9, y: 40 }}
               className="bg-white rounded-[64px] shadow-[0_48px_120px_-24px_rgba(0,0,0,0.3)] w-full max-w-5xl overflow-hidden border border-white relative shadow-indigo-100/50"
             >
                {/* Hero Gradient Header */}
                <div className="h-48 bg-indigo-600 relative overflow-hidden p-12 flex justify-between items-start">
                   <div className="absolute top-0 right-0 w-96 h-96 bg-white/10 rounded-full blur-[80px] -mr-32 -mt-32" />
                   <div className="absolute bottom-0 left-0 w-64 h-64 bg-indigo-400/20 rounded-full blur-[60px] -ml-32 -mb-32" />
                   
                   <div className="relative z-10 flex items-center gap-10">
                      <div className="w-24 h-24 bg-white/20 backdrop-blur-md rounded-[32px] border border-white/30 flex items-center justify-center text-white shadow-2xl">
                         <SparklesIcon className="w-10 h-10" />
                      </div>
                      <div>
                         <div className="flex items-center gap-3 mb-2">
                            <span className="px-4 py-1 bg-white/20 backdrop-blur-md rounded-full text-[10px] font-black text-white uppercase tracking-widest border border-white/20">Audit Inspect</span>
                            <div className="w-2 h-2 bg-emerald-400 rounded-full animate-ping" />
                         </div>
                         <h2 className="text-4xl font-black text-white thai-text tracking-tighter uppercase leading-none">{selectedProduct.name_th}</h2>
                      </div>
                   </div>

                   <button onClick={()=>setShowProductDetail(false)} className="relative z-10 w-16 h-16 bg-white/10 hover:bg-white/20 rounded-full flex items-center justify-center text-white transition-all backdrop-blur-md border border-white/20">
                      <XIcon className="w-7 h-7" />
                   </button>
                </div>

                {/* Content Matrix */}
                <div className="p-16 grid grid-cols-1 lg:grid-cols-3 gap-12">
                   <div className="lg:col-span-2 space-y-12">
                      <section>
                         <h5 className="text-[10px] font-black text-indigo-500 uppercase tracking-[0.4em] mb-6 flex items-center gap-3">
                            <div className="w-1 h-4 bg-indigo-600 rounded-full" />
                            Technical Specifications
                         </h5>
                         <div className="grid grid-cols-2 gap-6">
                            {[
                               { label: 'Merchant SKU', value: selectedProduct.sku || 'UNREADABLE' },
                               { label: 'Global Brand', value: selectedProduct.brand || 'GENERIC' },
                               { label: 'Model Tier', value: selectedProduct.model || 'STANDARD' },
                               { label: 'Taxonomy Category', value: selectedProduct.category?.name_th || 'PENDING' }
                            ].map((item, i) => (
                               <div key={i} className="p-8 bg-slate-50 rounded-[40px] border border-slate-100 hover:bg-white hover:shadow-xl transition-all group">
                                  <p className="text-[9px] font-black text-slate-400 uppercase tracking-widest mb-2 group-hover:text-indigo-400">{item.label}</p>
                                  <p className="text-lg font-black text-slate-800 thai-text tracking-tight italic">{item.value}</p>
                               </div>
                            ))}
                         </div>
                      </section>

                      <section>
                         <h5 className="text-[10px] font-black text-indigo-500 uppercase tracking-[0.4em] mb-6">Semantic Narrative</h5>
                         <div className="p-10 bg-slate-900 rounded-[48px] text-slate-300 text-lg font-medium leading-relaxed thai-text relative overflow-hidden">
                            <ActivityIcon className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64 text-white/5 pointer-events-none" />
                            "{selectedProduct.description || 'No descriptive metadata captured for this entity.'}"
                         </div>
                      </section>
                   </div>

                   <div className="space-y-12">
                      <section className="bg-indigo-50/50 p-10 rounded-[56px] border border-indigo-100 flex flex-col justify-between min-h-[320px]">
                         <div>
                            <p className="text-[10px] font-black text-indigo-400 uppercase tracking-widest mb-6">Registry Value</p>
                            <div className="flex items-baseline gap-2">
                               <span className="text-5xl font-black text-indigo-600 tracking-tighter">฿{selectedProduct.price?.toLocaleString()}</span>
                               <span className="text-xs font-bold text-indigo-300 uppercase italic">THB</span>
                            </div>
                         </div>
                         
                         <div className="pt-8 border-t border-indigo-100/50 space-y-4">
                            <div className="flex items-center justify-between">
                               <span className="text-[10px] font-black text-indigo-400 uppercase tracking-widest">Pricing Quality</span>
                               <div className="flex gap-1">
                                  {[1,2,3,4,5].map(s=><div key={s} className="w-1.5 h-4 bg-indigo-600 rounded-full" />)}
                               </div>
                            </div>
                         </div>
                      </section>

                      {selectedProduct.status === 'pending' && (
                        <section className="space-y-4">
                           <button 
                             onClick={()=>handleProductReview(selectedProduct.id, 'approved')}
                             className="w-full py-6 bg-slate-900 text-white rounded-[32px] font-black uppercase tracking-[0.3em] hover:bg-emerald-600 hover:shadow-2xl transition-all shadow-xl shadow-slate-200 flex items-center justify-center gap-4 group"
                           >
                              <CheckCircleIcon className="w-6 h-6 group-hover:rotate-12 transition-all" />
                              Commit Approval
                           </button>
                           <button 
                             onClick={()=>handleProductReview(selectedProduct.id, 'rejected')}
                             className="w-full py-6 bg-white border-2 border-rose-100 text-rose-500 rounded-[32px] font-black uppercase tracking-[0.3em] hover:bg-rose-50 hover:border-rose-300 transition-all flex items-center justify-center gap-4 group"
                           >
                              <XCircleIcon className="w-6 h-6 group-hover:scale-90 transition-all" />
                              Decline Entry
                           </button>
                        </section>
                      )}
                   </div>
                </div>
             </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  )
}
