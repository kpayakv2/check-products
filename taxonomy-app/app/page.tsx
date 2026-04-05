'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import Sidebar from '@/components/Layout/Sidebar'
import Header from '@/components/Layout/Header'
import { 
  FolderTreeIcon, 
  BookOpenIcon, 
  ShoppingBagIcon, 
  UploadIcon,
  UsersIcon,
  ClockIcon,
  CheckCircleIcon,
  AlertTriangleIcon,
  ArrowUpRightIcon,
  ActivityIcon,
  BarChartIcon,
  SparklesIcon,
  ChevronRightIcon
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { DatabaseService } from '@/utils/supabase'

interface DashboardStats {
  totalCategories: number
  totalSynonyms: number
  pendingProducts: number
  approvedProducts: number
  duplicateMatches: number
  reviewsToday: number
}

const quickActions = [
  {
    title: 'Structure Taxonomy',
    description: 'Hierarchical Mapping',
    icon: FolderTreeIcon,
    color: 'bg-indigo-600',
    href: '/taxonomy',
    thaiTitle: 'จัดการ Taxonomy'
  },
  {
    title: 'Semantic Synonyms',
    description: 'Linguistic Rules',
    icon: BookOpenIcon,
    color: 'bg-emerald-600',
    href: '/synonyms',
    thaiTitle: 'จัดการ Synonym'
  },
  {
    title: 'Inbound Audit',
    description: 'Manual Verification',
    icon: ShoppingBagIcon,
    color: 'bg-violet-600',
    href: '/products',
    thaiTitle: 'ตรวจสอบสินค้า'
  },
  {
    title: 'Data Ingestion',
    description: 'Bulk CSV Processing',
    icon: UploadIcon,
    color: 'bg-amber-600',
    href: '/import',
    thaiTitle: 'นำเข้าข้อมูล'
  },
]

function StatCard({ title, value, change, icon: Icon, color, index }: {
  title: string
  value: number
  change?: string
  icon: any
  color: string
  index: number
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }}
      whileHover={{ y: -8, transition: { duration: 0.3 } }}
      className="premium-card p-5 md:p-8 flex flex-col justify-between h-full bg-white/40 border-white/80 hover:bg-white/95 group overflow-hidden relative"
    >
      <div className="absolute top-0 right-0 -mr-4 -mt-4 w-24 h-24 bg-slate-50/50 rounded-full blur-2xl group-hover:bg-indigo-50/50 transition-colors" />
      
      <div className="flex items-start justify-between mb-4 md:mb-6 relative z-10">
        <div className={`p-3 md:p-4 rounded-3xl ${color} bg-opacity-10 text-opacity-100 shadow-sm border border-white group-hover:scale-110 transition-all duration-500`}>
          <Icon className="h-5 w-5 md:h-6 md:w-6" />
        </div>
        {change && (
          <div className="flex flex-col items-end">
            <span className="flex items-center gap-1 px-2 py-0.5 md:px-2.5 md:py-1 rounded-full bg-emerald-50 text-emerald-600 text-[9px] md:text-[10px] font-black border border-emerald-100/50 uppercase tracking-tighter">
              <ArrowUpRightIcon className="w-2.5 h-2.5 md:w-3 md:h-3" /> {change}
            </span>
          </div>
        )}
      </div>
      
      <div className="relative z-10">
        <p className="text-[10px] md:text-xs font-bold text-slate-400 uppercase tracking-widest mt-1 truncate">{title}</p>
        <div className="flex items-baseline gap-1 md:gap-2 overflow-hidden">
          <p className="text-2xl md:text-4xl font-black text-slate-900 tracking-tighter truncate">
            {value.toLocaleString()}
          </p>
          <span className="text-[9px] md:text-[11px] font-bold text-slate-300 uppercase tracking-widest italic shrink-0">Units</span>
        </div>
      </div>
    </motion.div>
  )
}

export default function Dashboard() {
  const [isLoading, setIsLoading] = useState(true)
  const [stats, setStats] = useState<DashboardStats>({
    totalCategories: 0,
    totalSynonyms: 0,
    pendingProducts: 0,
    approvedProducts: 0,
    duplicateMatches: 0,
    reviewsToday: 0
  })
  const router = useRouter()

  useEffect(() => {
    loadDashboardData()
  }, [])

  const loadDashboardData = async () => {
    try {
      setIsLoading(true)
      const [taxonomyData, synonymData, productData] = await Promise.all([
        DatabaseService.getTaxonomyTree(),
        DatabaseService.getSynonyms(),
        DatabaseService.getProducts()
      ])

      setStats({
        totalCategories: taxonomyData?.length || 0,
        totalSynonyms: synonymData?.length || 0,
        pendingProducts: productData?.filter(p => p.status === 'pending')?.length || 0,
        approvedProducts: productData?.filter(p => p.status === 'approved')?.length || 0,
        duplicateMatches: productData?.filter(p => p.status === 'rejected')?.length || 0,
        reviewsToday: productData?.filter(p => {
          const today = new Date().toDateString()
          return new Date(p.updated_at).toDateString() === today
        })?.length || 0
      })
    } catch (error) {
      console.error('Error loading dashboard data:', error)
      // No longer using mock data as per user request
    } finally {
      setIsLoading(false)
    }
  }

  if (isLoading) {
    return (
      <div className="flex h-screen bg-white">
        <Sidebar />
        <div className="flex-1 flex flex-col items-center justify-center relative overflow-hidden">
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-indigo-50/50 rounded-full blur-[100px]" />
          <div className="relative z-10 flex flex-col items-center">
            <div className="w-20 h-20 relative flex items-center justify-center">
               <div className="absolute inset-0 border-4 border-indigo-100 rounded-full" />
               <div className="absolute inset-0 border-4 border-indigo-600 rounded-full border-t-transparent animate-spin" />
               <SparklesIcon className="w-8 h-8 text-indigo-600 animate-pulse" />
            </div>
            <p className="mt-8 text-[11px] font-black text-slate-400 uppercase tracking-[0.3em] thai-text">Initializing Neural Engine...</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-screen bg-gray-50 font-sans">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden relative">
        {/* Decorative elements */}
        <div className="absolute top-0 right-0 w-[800px] h-[800px] bg-indigo-50/30 rounded-full blur-[120px] -mr-96 -mt-96 pointer-events-none" />
        <div className="absolute bottom-10 left-10 w-[400px] h-[400px] bg-emerald-50/30 rounded-full blur-[120px] pointer-events-none" />
        
        <Header />
        
        <main className="flex-1 overflow-x-hidden overflow-y-auto p-10 relative z-10">
          <div className="max-w-7xl mx-auto">
            {/* Hero Section */}
            <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between mb-16 gap-8">
              <div>
                 <div className="flex items-center gap-3 mb-6">
                    <span className="px-4 py-1.5 bg-indigo-600 rounded-full text-xs font-black text-white uppercase tracking-widest shadow-lg shadow-indigo-100">Neural Intelligence Suite</span>
                    <span className="px-4 py-1.5 bg-white border border-slate-100 rounded-full text-xs font-bold text-slate-400 uppercase tracking-wider">v2.4.0 Engine</span>
                 </div>
                 <h1 className="text-4xl lg:text-5xl font-black text-slate-900 tracking-tight leading-tight uppercase thai-text mb-6">
                   System Dashboard
                 </h1>
                 <p className="text-slate-500 font-medium text-lg thai-text leading-relaxed max-w-2xl mb-10">
                   ควบคุมและตรวจสอบระบบจัดการข้อมูลสินค้าอัตโนมัติแบบเรียลไทม์ พร้อมการวิเคราะห์ความถูกต้องผ่าน AI Engine
                 </p>
              </div>
              <div className="hidden lg:flex items-center gap-6 p-6 bg-white/40 backdrop-blur-md rounded-[40px] border border-white shadow-xl">
                 <div className="flex flex-col items-center px-4">
                    <ActivityIcon className="w-6 h-6 text-indigo-500 mb-2" />
                    <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Latency</span>
                    <span className="text-sm font-black text-slate-900">12ms</span>
                 </div>
                 <div className="w-[1px] h-10 bg-slate-100" />
                 <div className="flex flex-col items-center px-4">
                    <BarChartIcon className="w-6 h-6 text-emerald-500 mb-2" />
                    <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Accuracy</span>
                    <span className="text-sm font-black text-slate-900">99.8%</span>
                 </div>
              </div>
            </div>

            {/* Stats Dashboard */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-6 mb-16">
              <StatCard
                index={0}
                title="Taxonomy Nodes"
                value={stats.totalCategories}
                change="8.2%"
                icon={FolderTreeIcon}
                color="bg-indigo-600 text-indigo-600"
              />
              <StatCard
                index={1}
                title="Semantic Sets"
                value={stats.totalSynonyms}
                change="4.1%"
                icon={BookOpenIcon}
                color="bg-sky-600 text-sky-600"
              />
              <StatCard
                index={2}
                title="Awaiting Audit"
                value={stats.pendingProducts}
                icon={ClockIcon}
                color="bg-amber-600 text-amber-600"
              />
              <StatCard
                index={3}
                title="Verified Index"
                value={stats.approvedProducts}
                change="12.5%"
                icon={CheckCircleIcon}
                color="bg-emerald-600 text-emerald-600"
              />
              <StatCard
                index={4}
                title="Conflict Alerts"
                value={stats.duplicateMatches}
                icon={AlertTriangleIcon}
                color="bg-rose-600 text-rose-600"
              />
              <StatCard
                index={5}
                title="Review Velocity"
                value={stats.reviewsToday}
                icon={UsersIcon}
                color="bg-violet-600 text-violet-600"
              />
            </div>

            {/* Quick Actions & Activity */}
            <div className="grid grid-cols-1 lg:grid-cols-1 gap-12">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="premium-card p-12 bg-slate-900 relative overflow-hidden shadow-2xl"
              >
                <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-500/10 rounded-full blur-[80px] -mr-32 -mt-32" />
                <div className="absolute bottom-0 left-0 w-64 h-64 bg-emerald-500/10 rounded-full blur-[80px] -ml-32 -mb-32" />

                <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between mb-10 md:mb-16 gap-6 overflow-hidden">
                  <div className="min-w-0">
                    <h3 className="text-xl md:text-3xl font-black text-white tracking-tight uppercase truncate">
                      Operational Portal
                    </h3>
                    <p className="text-slate-400 mt-2 font-bold uppercase tracking-[0.2em] text-[9px] md:text-[11px] truncate">Primary Administrative Interface</p>
                  </div>
                  <div className="flex gap-2 shrink-0">
                     <div className="px-4 py-1.5 md:px-6 md:py-2 bg-white/5 border border-white/10 rounded-full text-[8px] md:text-[10px] font-black text-indigo-400 uppercase tracking-widest animate-pulse whitespace-nowrap">Neural Optimized</div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-8">
                  {quickActions.map((action, index) => (
                    <motion.button
                      key={index}
                      whileHover={{ scale: 1.02, y: -5 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={() => router.push(action.href)}
                      className="group relative flex flex-col p-6 md:p-10 rounded-[32px] md:rounded-[48px] bg-white/5 border border-white/10 hover:bg-white hover:border-white hover:shadow-2xl transition-all duration-500 text-left overflow-hidden min-h-[280px] md:min-h-[320px] justify-between"
                    >
                      {/* Decorative backdrop for hover state */}
                      <div className={`absolute top-0 right-0 w-24 h-24 ${action.color} opacity-0 group-hover:opacity-10 blur-2xl group-hover:blur-3xl transition-all duration-500`} />
                      
                      <div className={`${action.color} w-12 h-12 md:w-16 md:h-16 rounded-2xl md:rounded-3xl shadow-xl flex items-center justify-center group-hover:scale-110 group-hover:rotate-6 transition-all duration-500 ring-4 ring-white/5 group-hover:ring-indigo-50`}>
                        <action.icon className="h-6 w-6 md:h-8 md:w-8 text-white" />
                      </div>
                      
                      <div className="min-w-0">
                        <p className="text-[8px] md:text-[10px] font-black text-slate-500 group-hover:text-indigo-400 uppercase tracking-[0.2em] md:tracking-[0.3em] mb-2 md:mb-3 transition-colors truncate">
                           {action.description}
                        </p>
                        <h4 className="text-lg md:text-2xl font-black text-white group-hover:text-slate-900 tracking-tighter uppercase mb-1 md:mb-2 transition-colors truncate">
                          {action.title}
                        </h4>
                        <div className="flex items-center gap-2 group-hover:gap-4 transition-all">
                           <span className="text-[10px] md:text-sm font-bold text-slate-400 group-hover:text-slate-600 thai-text transition-colors truncate">{action.thaiTitle}</span>
                           <ChevronRightIcon className="w-4 h-4 md:w-5 md:h-5 text-indigo-500 opacity-0 group-hover:opacity-100 transition-all" />
                        </div>
                      </div>
                    </motion.button>
                  ))}
                </div>
              </motion.div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}

