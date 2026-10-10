'use client'

import { useState, useEffect, useRef } from 'react'
import { useRouter } from 'next/navigation'
import { BellIcon, SearchIcon, UserIcon, LogOutIcon, SettingsIcon } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { FOCUS_RING } from '@/components/ui/Modal'

interface HeaderProps {
  title?: string
  subtitle?: string
}

export default function Header({ title = 'แดชบอร์ด', subtitle }: HeaderProps) {
  const router = useRouter()
  const [showUserMenu, setShowUserMenu] = useState(false)
  const [showNotifications, setShowNotifications] = useState(false)

  const notifButtonRef = useRef<HTMLButtonElement | null>(null)
  const userMenuButtonRef = useRef<HTMLButtonElement | null>(null)

  // Esc key listener to close open dropdowns and restore focus
  useEffect(() => {
    if (!showUserMenu && !showNotifications) return

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        if (showNotifications) {
          setShowNotifications(false)
          notifButtonRef.current?.focus()
        }
        if (showUserMenu) {
          setShowUserMenu(false)
          userMenuButtonRef.current?.focus()
        }
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [showUserMenu, showNotifications])

  // ระบบนี้ไม่มีบัญชีผู้ใช้ — "ออกจากระบบ" แปลว่าลบคุกกี้ปลดล็อกที่ทุกคนใช้ร่วมกันทิ้ง
  // แล้วพากลับไปหน้ากรอกรหัส ดู CLAUDE.md → Authentication
  const handleLogout = async () => {
    await fetch('/api/lock', { method: 'POST' })
    router.push('/unlock')
    router.refresh()
  }

  const notifications = [
    {
      id: 1,
      title: 'สินค้าใหม่รอการตรวจสอบ',
      message: 'มีสินค้าใหม่ 5 รายการรอการอนุมัติ',
      time: '5 นาทีที่แล้ว',
      type: 'info'
    },
    {
      id: 2,
      title: 'Synonym ใหม่ถูกเพิ่ม',
      message: 'มีการเพิ่ม synonym ใหม่ในหมวด "อิเล็กทรอนิกส์"',
      time: '1 ชั่วโมงที่แล้ว',
      type: 'success'
    },
    {
      id: 3,
      title: 'การจับคู่สินค้าซ้ำ',
      message: 'พบสินค้าที่อาจซ้ำกัน 3 คู่',
      time: '2 ชั่วโมงที่แล้ว',
      type: 'warning'
    }
  ]

  return (
    <header className="bg-white/70 backdrop-blur-lg border-b border-slate-200/50 sticky top-0 z-30">
      <div className="px-8 py-5">
        <div className="flex items-center justify-between">
          {/* Title Section */}
          <div className="flex-1">
            <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight thai-text">
              {title}
            </h1>
            {subtitle && (
              <p className="text-xs font-bold text-indigo-500 uppercase tracking-widest mt-1.5 thai-text">
                {subtitle}
              </p>
            )}
          </div>

          {/* Actions Section */}
          <div className="flex items-center space-x-4">
            {/* Search */}
            <div className="relative hidden md:block">
              <label htmlFor="header-search-input" className="sr-only">
                ค้นหาข้อมูล
              </label>
              <SearchIcon className="absolute left-4 top-1/2 transform -translate-y-1/2 h-4 w-4 text-slate-400 pointer-events-none" aria-hidden="true" />
              <input
                id="header-search-input"
                type="search"
                placeholder="ค้นหาข้อมูล..."
                className={`pl-11 pr-4 py-2.5 w-72 text-sm bg-slate-100/50 border border-slate-200 rounded-xl focus:bg-white transition-all duration-300 thai-text ${FOCUS_RING}`}
              />
            </div>

            {/* Notifications */}
            <div className="relative">
              <button
                ref={notifButtonRef}
                onClick={() => setShowNotifications(!showNotifications)}
                aria-label={`การแจ้งเตือน ${notifications.length > 0 ? `มี ${notifications.length} รายการ` : 'ไม่มีรายการใหม่'}`}
                aria-expanded={showNotifications}
                className={`
                  relative p-2.5 rounded-xl transition-all duration-300 ${FOCUS_RING}
                  ${showNotifications ? 'bg-indigo-50 text-indigo-600' : 'text-slate-500 hover:bg-slate-100 hover:text-slate-700'}
                `}
              >
                <BellIcon className="h-5 w-5" aria-hidden="true" />
                {notifications.length > 0 && (
                  <span className="absolute top-1.5 right-1.5 h-4.5 min-w-[1.125rem] px-1 bg-rose-600 text-white text-xs font-black rounded-full flex items-center justify-center ring-2 ring-white">
                    {notifications.length}
                  </span>
                )}
              </button>

              {/* Notifications Dropdown */}
              <AnimatePresence>
                {showNotifications && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95, y: -10 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.95, y: -10 }}
                    className="absolute right-0 mt-2 w-80 bg-white rounded-xl shadow-premium border border-gray-200 z-50"
                  >
                    <div className="p-4 border-b border-gray-100">
                      <h2 className="text-base font-bold text-gray-900 thai-text">
                        การแจ้งเตือน
                      </h2>
                    </div>
                    <div className="max-h-96 overflow-y-auto">
                      {notifications.map((notification) => (
                        <div key={notification.id} className="p-4 border-b border-gray-50 hover:bg-gray-50 transition-colors duration-150">
                          <div className="flex items-start space-x-3">
                            <div className={`w-2 h-2 rounded-full mt-2 ${
                              notification.type === 'info' ? 'bg-indigo-500' :
                              notification.type === 'success' ? 'bg-emerald-500' :
                              'bg-amber-500'
                            }`} aria-hidden="true" />
                            <div className="flex-1">
                              <h3 className="text-sm font-semibold text-gray-900 thai-text">
                                {notification.title}
                              </h3>
                              <p className="text-sm text-gray-600 thai-text mt-1">
                                {notification.message}
                              </p>
                              <p className="text-xs text-gray-400 mt-2">
                                {notification.time}
                              </p>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                    <div className="p-4 border-t border-gray-100">
                      <button className={`w-full text-center text-sm text-indigo-600 hover:text-indigo-700 font-bold py-1 rounded-lg thai-text ${FOCUS_RING}`}>
                        ดูการแจ้งเตือนทั้งหมด
                      </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* User Menu */}
            <div className="relative">
              <button
                ref={userMenuButtonRef}
                onClick={() => setShowUserMenu(!showUserMenu)}
                aria-label="เมนูผู้ใช้งาน"
                aria-expanded={showUserMenu}
                className={`
                  flex items-center space-x-3 p-1.5 pr-4 rounded-2xl transition-all duration-300 ${FOCUS_RING}
                  ${showUserMenu ? 'bg-indigo-50 shadow-sm' : 'hover:bg-slate-50'}
                `}
              >
                <div className="w-9 h-9 bg-gradient-to-br from-indigo-500 to-indigo-600 rounded-xl flex items-center justify-center shadow-md shadow-indigo-200">
                  <UserIcon className="h-5 w-5 text-white" aria-hidden="true" />
                </div>
                <div className="hidden md:block text-left">
                  <p className="text-sm font-bold text-slate-900 thai-text leading-tight">
                    ผู้ดูแลระบบ
                  </p>
                  <p className="text-xs font-bold text-indigo-500 uppercase tracking-tight">
                    Admin
                  </p>
                </div>
              </button>

              {/* User Dropdown */}
              <AnimatePresence>
                {showUserMenu && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95, y: -10 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.95, y: -10 }}
                    className="absolute right-0 mt-2 w-48 bg-white rounded-xl shadow-premium border border-gray-200 z-50"
                  >
                    <div className="p-4 border-b border-gray-100">
                      <p className="text-sm font-bold text-gray-900 thai-text">
                        ผู้ดูแลระบบ
                      </p>
                      <p className="text-xs text-gray-500">
                        admin@company.com
                      </p>
                    </div>
                    <div className="py-2">
                      <button className={`flex items-center w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors duration-150 ${FOCUS_RING}`}>
                        <SettingsIcon className="h-4 w-4 mr-3 text-gray-400" aria-hidden="true" />
                        <span className="thai-text">ตั้งค่าบัญชี</span>
                      </button>
                      <button
                        onClick={handleLogout}
                        className={`flex items-center w-full px-4 py-2 text-sm text-rose-600 hover:bg-rose-50 transition-colors duration-150 ${FOCUS_RING}`}
                      >
                        <LogOutIcon className="h-4 w-4 mr-3 text-rose-500" aria-hidden="true" />
                        <span className="thai-text">ออกจากระบบ</span>
                      </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </div>

      {/* Click outside to close dropdowns */}
      {(showUserMenu || showNotifications) && (
        <div 
          className="fixed inset-0 z-20" 
          aria-hidden="true"
          onClick={() => {
            setShowUserMenu(false)
            setShowNotifications(false)
          }}
        />
      )}
    </header>
  )
}

