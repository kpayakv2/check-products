'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { 
  HomeIcon, 
  FolderTreeIcon, 
  BookOpenIcon, 
  ShoppingBagIcon,
  UploadIcon,
  BarChartIcon,
  SettingsIcon,
  MenuIcon,
  XIcon,
  LayersIcon,
  SparklesIcon,
  ZapIcon
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

const navigation = [
  {
    name: 'Dashboard',
    href: '/',
    icon: HomeIcon,
    description: 'System Overview',
    thai: 'แดชบอร์ด'
  },
  {
    name: 'Taxonomy',
    href: '/taxonomy',
    icon: FolderTreeIcon,
    description: 'Hierarchical Structure',
    thai: 'จัดการ Taxonomy'
  },
  {
    name: 'Synonyms',
    href: '/synonyms',
    icon: BookOpenIcon,
    description: 'Semantic Mapping',
    thai: 'จัดการ Synonym'
  },
  {
    name: 'Product Audit',
    href: '/products',
    icon: ShoppingBagIcon,
    description: 'Quality Control',
    thai: 'ตรวจสอบสินค้า'
  },
  {
    name: 'Deduplication',
    href: '/deduplication',
    icon: LayersIcon,
    description: 'Conflict Resolution',
    thai: 'คัดกรองสินค้าซ้ำ'
  },
  {
    name: 'Data Import',
    href: '/import',
    icon: UploadIcon,
    description: 'Bulk Intake',
    thai: 'นำเข้าข้อมูล'
  },
  {
    name: 'Reports',
    href: '/reports',
    icon: BarChartIcon,
    description: 'Performance Analytics',
    thai: 'รายงานวิเคราะห์'
  },
  {
    name: 'Settings',
    href: '/settings',
    icon: SettingsIcon,
    description: 'Configuration',
    thai: 'ตั้งค่า'
  },
]

interface SidebarProps {
  className?: string
}

export default function Sidebar({ className = '' }: SidebarProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [isDesktop, setIsDesktop] = useState(false)
  const pathname = usePathname()

  useEffect(() => {
    // Initial check
    setIsDesktop(window.innerWidth >= 1024)

    const handleResize = () => {
      const desktop = window.innerWidth >= 1024
      setIsDesktop(desktop)
      if (desktop) setIsOpen(false)
    }

    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  const sidebarVariants = {
    open: {
      x: 0,
      transition: {
        type: "spring",
        stiffness: 300,
        damping: 30
      }
    },
    closed: {
      x: "-100%",
      transition: {
        type: "spring",
        stiffness: 300,
        damping: 30
      }
    }
  }

  return (
    <>
      {/* Mobile menu button */}
      <button
        onClick={() => setIsOpen(true)}
        className="lg:hidden fixed top-6 left-6 z-50 p-3 rounded-2xl bg-white shadow-2xl border border-slate-100"
      >
        <MenuIcon className="h-6 w-6 text-slate-600" />
      </button>

      {/* Overlay for mobile */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setIsOpen(false)}
            className="lg:hidden fixed inset-0 bg-slate-900/40 backdrop-blur-md z-40"
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <motion.aside
        variants={sidebarVariants}
        initial={isDesktop ? "open" : "closed"}
        animate={isDesktop ? "open" : (isOpen ? "open" : "closed")}
        className={`
          fixed lg:static inset-y-0 left-0 z-50 w-80 bg-white border-r border-slate-100
          lg:translate-x-0 lg:block
          ${className}
        `}
      >
        <div className="flex flex-col h-full relative overflow-hidden">
          {/* Decorative backdrop */}
          <div className="absolute top-0 left-0 w-full h-64 bg-gradient-to-b from-indigo-50/50 to-transparent pointer-events-none" />
          
          {/* Header */}
          <div className="flex items-center justify-between p-8 relative z-10">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 bg-indigo-600 rounded-[18px] flex items-center justify-center shadow-xl shadow-indigo-100 ring-4 ring-indigo-50">
                <SparklesIcon className="h-6 w-6 text-white" />
              </div>
              <div className="flex flex-col">
                <h1 className="text-xl font-black text-slate-900 tracking-tighter uppercase leading-none">
                  Neural
                </h1>
                <p className="text-xs font-bold text-indigo-500 uppercase tracking-widest leading-relaxed mt-1">
                  Taxonomy Engine
                </p>
              </div>
            </div>
            
            <button
              onClick={() => setIsOpen(false)}
              className="lg:hidden p-2 rounded-xl hover:bg-slate-100 transition-colors"
            >
              <XIcon className="h-5 w-5 text-slate-500" />
            </button>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-6 py-8 space-y-2 overflow-y-auto relative z-10 custom-scrollbar">
             <p className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-6 px-4">Core Orchestration</p>
            {navigation.map((item) => {
              const isActive = pathname === item.href
              const Icon = item.icon

              return (
                <Link
                  key={item.name}
                  href={item.href}
                  onClick={() => setIsOpen(false)}
                  className={`
                    group flex items-center px-4 py-4 rounded-[24px] transition-all duration-500 relative
                    ${isActive 
                      ? 'bg-indigo-600 text-white shadow-2xl shadow-indigo-200' 
                      : 'text-slate-500 hover:bg-slate-50 hover:text-slate-900'
                    }
                  `}
                >
                  <div className={`
                    p-2 rounded-xl mr-4 transition-all duration-500
                    ${isActive ? 'bg-white/10' : 'bg-transparent'}
                  `}>
                    <Icon 
                      className={`
                        h-5 w-5 transition-all duration-500
                        ${isActive ? 'text-white' : 'text-slate-400 group-hover:text-indigo-600 group-hover:scale-110 group-hover:rotate-6'}
                      `} 
                    />
                  </div>
                  <div className="flex-1">
                    <p className={`text-[11px] font-bold uppercase tracking-wider leading-relaxed mb-1 transition-all ${isActive ? 'text-indigo-100' : 'text-slate-400'}`}>
                       {item.description}
                    </p>
                    <div className={`font-black text-sm tracking-tight transition-all ${isActive ? 'text-white' : 'text-slate-700'}`}>
                      {item.name} <span className="thai-text opacity-70 ml-1 font-bold group-hover:opacity-100 transition-opacity">({item.thai})</span>
                    </div>
                  </div>
                  
                  {isActive && (
                    <motion.div 
                      layoutId="sidebar-active-dot"
                      className="w-2 h-2 bg-white rounded-full ml-2 shadow-[0_0_8px_rgba(255,255,255,0.8)]" 
                    />
                  )}
                </Link>
              )
            })}
          </nav>

          {/* Footer Card */}
          <div className="p-8 relative z-10">
            <div className="premium-card p-6 bg-slate-900 border-none shadow-2xl relative overflow-hidden group">
               <div className="absolute top-0 right-0 w-24 h-24 bg-indigo-500/10 rounded-full blur-2xl group-hover:bg-indigo-500/20 transition-all" />
               <div className="flex items-center gap-4 relative z-10">
                  <div className="w-10 h-10 bg-white/5 rounded-xl border border-white/10 flex items-center justify-center">
                    <ZapIcon className="w-5 h-5 text-indigo-400 animate-pulse" />
                  </div>
                  <div>
                    <p className="text-[9px] font-black text-slate-500 uppercase tracking-widest leading-none mb-1">Service Status</p>
                    <p className="text-xs font-black text-white italic tracking-widest leading-none">All Systems Green</p>
                  </div>
               </div>
               
               <div className="mt-6 pt-6 border-t border-white/5 flex items-center justify-between gap-2 overflow-hidden">
                  <div className="flex flex-col min-w-0">
                     <span className="text-[8px] font-black text-slate-500 uppercase tracking-widest truncate">Build Signature</span>
                     <span className="text-[10px] font-bold text-slate-400 italic truncate">phayak-nexus-v1.2</span>
                  </div>
                  <div className="w-2 h-2 bg-emerald-500 rounded-full animate-ping shrink-0" />
               </div>
            </div>
          </div>
        </div>
      </motion.aside>
    </>
  )
}
