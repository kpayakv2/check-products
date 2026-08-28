'use client'

import { useState } from 'react'
import Sidebar from '@/components/Layout/Sidebar'
import Header from '@/components/Layout/Header'
import VerifyTab from '@/components/data-quality/VerifyTab'
import DeduplicationTab from '@/components/data-quality/DeduplicationTab'
import AutoLearnTab from '@/components/data-quality/AutoLearnTab'
import RecheckTab from '@/components/data-quality/RecheckTab'
import { LayersIcon, TagIcon, SparklesIcon, DatabaseIcon, ScanSearchIcon } from 'lucide-react'

export default function DataQualityCenter() {
  const [activeTab, setActiveTab] = useState<'dedup' | 'verify' | 'recheck' | 'autolearn'>('dedup')

  return (
    <div className="flex h-screen bg-[#F8FAFC]">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        
        {/* Navigation Tabs Header */}
        <div className="bg-white border-b border-slate-200 px-8 py-4 z-10 relative shadow-sm">
          <div className="max-w-7xl mx-auto flex flex-col gap-6">
            <div>
              <h1 className="text-3xl font-black text-slate-900 tracking-tight flex items-center gap-3 thai-text uppercase">
                <DatabaseIcon className="w-8 h-8 text-indigo-600" />
                Data Quality Center
              </h1>
              <p className="text-slate-500 font-medium mt-1 thai-text">ศูนย์ตรวจสอบคุณภาพข้อมูล: คัดกรองของซ้ำ จัดหมวดหมู่ และเรียนรู้อัตโนมัติ</p>
            </div>
            
            {/* แท็บเลื่อนในกรอบตัวเองเมื่อจอแคบ ไม่ให้ดันทั้งหน้าจนเลื่อนแนวนอน */}
            <div className="flex gap-4 p-1.5 bg-slate-100/80 rounded-[20px] w-fit max-w-full overflow-x-auto">
              <button 
                onClick={() => setActiveTab('dedup')}
                className={`px-8 py-3 rounded-[16px] text-sm font-black transition-all flex items-center gap-2 thai-text uppercase whitespace-nowrap shrink-0 ${
                  activeTab === 'dedup' ? 'bg-white text-indigo-600 shadow-md border border-slate-200/50' : 'text-slate-500 hover:text-slate-700 hover:bg-slate-200/50'
                }`}
              >
                <LayersIcon className="w-4 h-4" />
                เช็คข้อมูลซ้ำ (Deduplication)
              </button>
              <button 
                onClick={() => setActiveTab('verify')}
                className={`px-8 py-3 rounded-[16px] text-sm font-black transition-all flex items-center gap-2 thai-text uppercase whitespace-nowrap shrink-0 ${
                  activeTab === 'verify' ? 'bg-white text-indigo-600 shadow-md border border-slate-200/50' : 'text-slate-500 hover:text-slate-700 hover:bg-slate-200/50'
                }`}
              >
                <TagIcon className="w-4 h-4" />
                ตรวจสอบหมวดหมู่ (Verify)
              </button>
              <button
                onClick={() => setActiveTab('recheck')}
                className={`px-8 py-3 rounded-[16px] text-sm font-black transition-all flex items-center gap-2 thai-text uppercase whitespace-nowrap shrink-0 ${
                  activeTab === 'recheck' ? 'bg-white text-indigo-600 shadow-md border border-slate-200/50' : 'text-slate-500 hover:text-slate-700 hover:bg-slate-200/50'
                }`}
                data-testid="tab-recheck"
              >
                <ScanSearchIcon className="w-4 h-4" />
                ตรวจซ้ำของเก่า (Recheck)
              </button>
              <button
                onClick={() => setActiveTab('autolearn')}
                className={`px-8 py-3 rounded-[16px] text-sm font-black transition-all flex items-center gap-2 thai-text uppercase whitespace-nowrap shrink-0 ${
                  activeTab === 'autolearn' ? 'bg-white text-indigo-600 shadow-md border border-slate-200/50' : 'text-slate-500 hover:text-slate-700 hover:bg-slate-200/50'
                }`}
              >
                <SparklesIcon className="w-4 h-4" />
                ระบบเรียนรู้ AI (Auto-learn)
              </button>
            </div>
          </div>
        </div>

        {/* Tab Content */}
        <div className="flex-1 overflow-hidden relative bg-gray-50/50">
          {activeTab === 'dedup' && <DeduplicationTab onComplete={() => setActiveTab('verify')} />}
          {activeTab === 'verify' && <VerifyTab />}
          {activeTab === 'recheck' && (
            <div className="h-full overflow-y-auto px-8 py-6">
              <div className="max-w-5xl mx-auto">
                <RecheckTab />
              </div>
            </div>
          )}
          {activeTab === 'autolearn' && <AutoLearnTab />}
        </div>
      </div>
    </div>
  )
}
