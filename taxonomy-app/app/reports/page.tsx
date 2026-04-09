'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import Sidebar from '@/components/Layout/Sidebar'
import Header from '@/components/Layout/Header'
import { 
  BarChartIcon, 
  TrendingUpIcon, 
  TargetIcon, 
  ClockIcon, 
  CheckCircleIcon,
  AlertCircleIcon,
  ActivityIcon,
  PieChartIcon,
  ArrowUpRightIcon,
  ZapIcon,
  SparklesIcon,
  SearchIcon
} from 'lucide-react'

export default function ReportsPage() {
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Artificial delay for premium loading feel
    const timer = setTimeout(() => setIsLoading(false), 800)
    return () => clearTimeout(timer)
  }, [])

  if (isLoading) {
    return (
      <div data-testid="loading-indicator" className="flex h-screen bg-white">
        <Sidebar />
        <div className="flex-1 flex flex-col items-center justify-center">
          <div className="w-20 h-20 relative flex items-center justify-center">
             <div className="absolute inset-0 border-4 border-indigo-100 rounded-full" />
             <div className="absolute inset-0 border-4 border-indigo-600 rounded-full border-t-transparent animate-spin" />
             <BarChartIcon className="w-8 h-8 text-indigo-600 animate-pulse" />
          </div>
          <p className="mt-8 text-[11px] font-black text-slate-400 uppercase tracking-[0.3em] thai-text">Aggregating Neural Insights...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-screen bg-gray-50 font-sans">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden relative">
        {/* Decorative backdrop blobs */}
        <div className="absolute top-0 right-0 w-[800px] h-[800px] bg-indigo-50/40 rounded-full blur-[120px] -mr-96 -mt-96 pointer-events-none" />
        <div className="absolute bottom-10 left-10 w-[400px] h-[400px] bg-emerald-50/40 rounded-full blur-[120px] pointer-events-none" />
        
        <Header />
        
        <main className="flex-1 overflow-x-hidden overflow-y-auto p-10 relative z-10">
          <div className="max-w-7xl mx-auto">
            {/* Hero Section */}
            <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between mb-16 gap-8">
              <div>
                 <div className="flex items-center space-x-3 mb-6">
                    <div className="w-1.5 h-6 bg-indigo-600 rounded-full" />
                    <span className="text-xs font-bold text-slate-400 uppercase tracking-wider leading-relaxed">Intelligence Report Portal</span>
                 </div>
                 <h1 data-testid="reports-title" className="text-3xl lg:text-4xl font-black text-slate-900 tracking-tight uppercase thai-text mb-6">
                   Analytics Portal
                 </h1>
                 <p className="text-slate-500 font-medium text-lg thai-text leading-relaxed max-w-2xl mb-10">
                   ตรวจสอบความแม่นยำและประสิทธิภาพของระบบ AI ในการจัดหมวดหมู่สินค้าแบบเรียลไทม์
                 </p>
              </div>

              <div className="flex items-center gap-3">
                 <button className="px-8 py-3 bg-white border border-slate-100 rounded-full text-xs font-black text-slate-600 uppercase tracking-widest shadow-sm hover:shadow-md transition-all flex items-center gap-2 group">
                    <ZapIcon className="w-4 h-4 text-indigo-500 group-hover:scale-110" />
                    Live Metrics
                 </button>
              </div>
            </div>

            {/* Key KPI Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-16">
              {[
                { label: 'Overall Accuracy', value: '99.8%', trend: '+0.2%', icon: TargetIcon, color: 'indigo', testid: 'accuracy-card' },
                { label: 'Process Velocity', value: '1.2k', trend: '+14%', icon: ActivityIcon, color: 'emerald', testid: 'velocity-card' },
                { label: 'Backlog Status', value: '23', trend: '-12%', icon: ClockIcon, color: 'amber', testid: 'backlog-card' },
                { label: 'Verified Index', value: '8.9k', trend: '+5%', icon: CheckCircleIcon, color: 'sky', testid: 'verified-card' }
              ].map((kpi, i) => (
                <motion.div
                  key={i}
                  data-testid={kpi.testid}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.1 }}
                  className="premium-card p-10 bg-white/60 border-white/80 hover:bg-white hover:shadow-2xl transition-all duration-500 group overflow-hidden relative"
                >
                  <div className={`absolute top-0 right-0 p-8 text-${kpi.color}-500/10 transition-transform group-hover:rotate-12`}>
                    <kpi.icon className="w-24 h-24" />
                  </div>
                  
                  <div className="relative z-10">
                     <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">{kpi.label}</p>
                     <div className="flex items-baseline gap-3">
                        <h3 className="text-4xl font-black text-slate-900 tracking-tighter">{kpi.value}</h3>
                        <span className={`text-xs font-black px-2 py-0.5 rounded-full ${kpi.trend.startsWith('+') ? 'bg-emerald-50 text-emerald-600' : 'bg-rose-50 text-rose-600'}`}>
                           {kpi.trend}
                        </span>
                     </div>
                  </div>
                </motion.div>
              ))}
            </div>

            {/* Secondary Visuals Section */}
            <div className="grid grid-cols-1 lg:grid-cols-1 gap-12">
               <motion.div
                  initial={{ opacity: 0, scale: 0.98 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="premium-card p-12 bg-slate-900 relative overflow-hidden"
               >
                  <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-indigo-500/10 rounded-full blur-[100px] -mr-48 -mt-48" />
                  
                  <div className="relative z-10 flex flex-col md:flex-row justify-between mb-16 gap-8">
                     <div>
                        <h3 className="text-3xl font-black text-white tracking-tight uppercase">Neural Matching Heatmap</h3>
                        <p className="text-slate-400 mt-1 uppercase text-[10px] font-black tracking-[0.2em]">Data Harmonization Distribution</p>
                     </div>
                     <div className="flex items-center gap-6">
                        <div className="flex items-center gap-2">
                           <div className="w-2 h-2 rounded-full bg-indigo-500" />
                           <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">High Confidence</span>
                        </div>
                        <div className="flex items-center gap-2">
                           <div className="w-2 h-2 rounded-full bg-amber-500" />
                           <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Manual Required</span>
                        </div>
                     </div>
                  </div>

                  {/* Mock Chart Visualization */}
                  <div data-testid="heatmap-chart" className="h-[400px] w-full flex items-end justify-between gap-4 px-4 relative z-10 pt-10">
                     {[40, 70, 45, 90, 65, 85, 30, 95, 60, 75, 55, 80].map((h, i) => (
                        <div key={i} className="flex-1 flex flex-col items-center group cursor-pointer h-full justify-end">
                           <motion.div
                              initial={{ height: 0 }}
                              animate={{ height: `${h}%` }}
                              transition={{ delay: i * 0.05, duration: 1, ease: 'easeOut' }}
                              data-testid={`chart-bar-${i}`}
                              className={`w-full max-w-[40px] rounded-t-2xl relative transition-all duration-500 group-hover:brightness-125 ${i % 3 === 0 ? 'bg-indigo-600 shadow-[0_0_20px_rgba(79,70,229,0.4)]' : 'bg-white/10 group-hover:bg-white/20'}`}
                           >
                              <div className="absolute -top-10 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity bg-white text-slate-900 px-3 py-1 rounded-lg text-[10px] font-black shadow-xl">
                                 {h}%
                              </div>
                           </motion.div>
                           <p className="mt-4 text-[9px] font-black text-slate-500 uppercase tracking-tighter">Month {i + 1}</p>
                        </div>
                     ))}
                  </div>
               </motion.div>

               <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
                  <div data-testid="activity-log" className="premium-card p-12 bg-white/40 border-white shadow-xl">
                     <div className="flex items-center justify-between mb-10">
                        <h4 className="text-xl font-black text-slate-900 uppercase tracking-tight">Recent Activity Log</h4>
                        <button className="text-[10px] font-black text-indigo-500 uppercase tracking-widest hover:underline">Full Audit</button>
                     </div>
                     <div className="space-y-8">
                        {[
                           { action: 'Auto-Classification', target: 'Beverage Category', time: '2m ago', icon: SparklesIcon, color: 'indigo' },
                           { action: 'Conflict Resolved', target: 'P&G SKU 458-1', time: '12m ago', icon: SearchIcon, color: 'emerald' },
                           { action: 'User Update', target: 'Synonym Rule #45', time: '45m ago', icon: TrendingUpIcon, color: 'sky' }
                        ].map((log, i) => (
                           <div key={i} className="flex items-center justify-between group">
                              <div className="flex items-center gap-6">
                                 <div className={`w-12 h-12 rounded-2xl bg-${log.color}-50 flex items-center justify-center text-${log.color}-600 border border-${log.color}-100 group-hover:scale-110 transition-transform`}>
                                    <log.icon className="w-5 h-5" />
                                 </div>
                                 <div>
                                    <h5 className="text-sm font-black text-slate-800 uppercase tracking-tight leading-none mb-1.5">{log.action}</h5>
                                    <p className="text-[11px] font-bold text-slate-400 uppercase tracking-widest">{log.target}</p>
                                 </div>
                              </div>
                              <span className="text-[10px] font-black text-slate-300 uppercase italic whitespace-nowrap">{log.time}</span>
                           </div>
                        ))}
                     </div>
                  </div>

                  <div className="premium-card p-12 bg-white/40 border-white shadow-xl flex flex-col justify-between">
                     <div>
                        <h4 className="text-xl font-black text-slate-900 uppercase tracking-tight mb-4">Neural Health Score</h4>
                        <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-10">Real-time system integrity check</p>
                        
                        <div className="relative w-48 h-48 mx-auto flex items-center justify-center">
                           <svg className="w-full h-full -rotate-90">
                              <circle cx="96" cy="96" r="88" stroke="currentColor" strokeWidth="12" fill="transparent" className="text-slate-100" />
                              <motion.circle 
                                 initial={{ strokeDasharray: '0 553' }}
                                 animate={{ strokeDasharray: '520 553' }}
                                 transition={{ duration: 2, ease: 'easeInOut' }}
                                 cx="96" cy="96" r="88" stroke="currentColor" strokeWidth="12" fill="transparent" strokeLinecap="round" className="text-indigo-600" 
                              />
                           </svg>
                           <div className="absolute inset-0 flex flex-col items-center justify-center">
                              <span className="text-5xl font-black text-slate-900 tracking-tighter">98</span>
                              <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Optimized</span>
                           </div>
                        </div>
                     </div>
                     <div className="pt-10 border-t border-slate-100 flex items-center justify-center gap-6">
                        <div className="flex items-center gap-2">
                           <div className="w-2 h-2 rounded-full bg-emerald-500" />
                           <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">Uptime: 99.9%</span>
                        </div>
                        <div className="flex items-center gap-2">
                           <div className="w-2 h-2 rounded-full bg-indigo-500" />
                           <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">Load: Normal</span>
                        </div>
                     </div>
                  </div>
               </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
