'use client'

import { useState } from 'react'
import Sidebar from '@/components/Layout/Sidebar'
import Header from '@/components/Layout/Header'
import VerifyTab from '@/components/data-quality/VerifyTab'
import DeduplicationTab from '@/components/data-quality/DeduplicationTab'
import AutoLearnTab from '@/components/data-quality/AutoLearnTab'
import RecheckTab from '@/components/data-quality/RecheckTab'
import PosExportTab from '@/components/data-quality/PosExportTab'
import { FOCUS_RING } from '@/components/ui/Modal'
import { LayersIcon, TagIcon, SparklesIcon, DatabaseIcon, ScanSearchIcon, UploadIcon } from 'lucide-react'

type TabType = 'dedup' | 'verify' | 'recheck' | 'autolearn' | 'posexport'

interface TabDef {
  id: TabType
  label: string
  icon: React.ComponentType<{ className?: string; 'aria-hidden'?: boolean | 'true' | 'false' }>
  testId?: string
}

const TABS: TabDef[] = [
  { id: 'dedup', label: 'เช็คข้อมูลซ้ำ (Deduplication)', icon: LayersIcon },
  { id: 'verify', label: 'ตรวจสอบหมวดหมู่ (Verify)', icon: TagIcon },
  { id: 'recheck', label: 'ตรวจซ้ำของเก่า (Recheck)', icon: ScanSearchIcon, testId: 'tab-recheck' },
  { id: 'autolearn', label: 'ระบบเรียนรู้ AI (Auto-learn)', icon: SparklesIcon },
  { id: 'posexport', label: 'ส่งออก POS', icon: UploadIcon, testId: 'tab-pos-export' },
]

export default function DataQualityCenter() {
  const [activeTab, setActiveTab] = useState<TabType>('dedup')

  const handleTabKeyDown = (e: React.KeyboardEvent, index: number) => {
    let nextIndex = index
    if (e.key === 'ArrowRight') {
      nextIndex = (index + 1) % TABS.length
    } else if (e.key === 'ArrowLeft') {
      nextIndex = (index - 1 + TABS.length) % TABS.length
    } else if (e.key === 'Home') {
      nextIndex = 0
    } else if (e.key === 'End') {
      nextIndex = TABS.length - 1
    } else {
      return
    }
    e.preventDefault()
    setActiveTab(TABS[nextIndex].id)
    const el = document.getElementById(`tab-${TABS[nextIndex].id}`)
    el?.focus()
  }

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
                <DatabaseIcon className="w-8 h-8 text-indigo-600" aria-hidden="true" />
                Data Quality Center
              </h1>
              <p className="text-slate-600 font-medium mt-1 thai-text">ศูนย์ตรวจสอบคุณภาพข้อมูล: คัดกรองของซ้ำ จัดหมวดหมู่ และเรียนรู้อัตโนมัติ</p>
            </div>
            
            {/* แท็บเลื่อนในกรอบตัวเองเมื่อจอแคบ ไม่ให้ดันทั้งหน้าจนเลื่อนแนวนอน */}
            <div 
              role="tablist"
              aria-label="เมนูศูนย์ตรวจสอบคุณภาพข้อมูล"
              className="flex gap-3 p-1.5 bg-slate-100/90 rounded-[20px] w-fit max-w-full overflow-x-auto"
            >
              {TABS.map((tab, idx) => {
                const Icon = tab.icon
                const isSelected = activeTab === tab.id
                return (
                  <button
                    key={tab.id}
                    id={`tab-${tab.id}`}
                    role="tab"
                    type="button"
                    aria-selected={isSelected}
                    aria-controls={`tabpanel-${tab.id}`}
                    tabIndex={isSelected ? 0 : -1}
                    onClick={() => setActiveTab(tab.id)}
                    onKeyDown={(e) => handleTabKeyDown(e, idx)}
                    data-testid={tab.testId}
                    className={`px-6 py-3 rounded-[16px] text-sm font-bold transition-all flex items-center gap-2 thai-text uppercase whitespace-nowrap shrink-0 ${FOCUS_RING} ${
                      isSelected
                        ? 'bg-white text-indigo-700 shadow-md border border-slate-200/80'
                        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
                    }`}
                  >
                    <Icon className="w-4 h-4" aria-hidden="true" />
                    {tab.label}
                  </button>
                )
              })}
            </div>
          </div>
        </div>

        {/* Tab Content */}
        <main
          role="tabpanel"
          id={`tabpanel-${activeTab}`}
          aria-labelledby={`tab-${activeTab}`}
          tabIndex={0}
          className={`flex-1 overflow-hidden relative bg-gray-50/50 ${FOCUS_RING}`}
        >
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
          {activeTab === 'posexport' && (
            <div className="h-full overflow-y-auto px-8 py-6">
              <div className="max-w-5xl mx-auto">
                <PosExportTab />
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  )
}
