'use client'

import { useState } from 'react'
import { BellIcon, SearchIcon, UserIcon, LogOutIcon, SettingsIcon } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

interface HeaderProps {
  title?: string
  subtitle?: string
}

export default function Header({ title = 'แดชบอร์ด', subtitle }: HeaderProps) {
  const [showUserMenu, setShowUserMenu] = useState(false)
  const [showNotifications, setShowNotifications] = useState(false)

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
              <SearchIcon className="absolute left-4 top-1/2 transform -translate-y-1/2 h-4 w-4 text-slate-400 group-focus-within:text-indigo-500 transition-colors" />
              <input
                type="text"
                placeholder="ค้นหาข้อมูล..."
                className="pl-11 pr-4 py-2.5 w-72 text-sm bg-slate-100/50 border border-transparent rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500/50 focus:bg-white transition-all duration-300 thai-text"
              />
            </div>

            {/* Notifications */}
            <div className="relative">
              <button
                onClick={() => setShowNotifications(!showNotifications)}
                className={`
                  relative p-2.5 rounded-xl transition-all duration-300 
                  ${showNotifications ? 'bg-indigo-50 text-indigo-600' : 'text-slate-400 hover:bg-slate-100 hover:text-slate-600'}
                `}
              >
                <BellIcon className="h-5 w-5" />
                {notifications.length > 0 && (
                  <span className="absolute top-1.5 right-1.5 h-4 w-4 bg-red-500 text-white text-[10px] font-bold rounded-full flex items-center justify-center ring-2 ring-white">
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
                      <h3 className="text-lg font-semibold text-gray-900 thai-text">
                        การแจ้งเตือน
                      </h3>
                    </div>
                    <div className="max-h-96 overflow-y-auto">
                      {notifications.map((notification) => (
                        <div key={notification.id} className="p-4 border-b border-gray-50 hover:bg-gray-50 transition-colors duration-150">
                          <div className="flex items-start space-x-3">
                            <div className={`w-2 h-2 rounded-full mt-2 ${
                              notification.type === 'info' ? 'bg-primary-500' :
                              notification.type === 'success' ? 'bg-success-500' :
                              'bg-warning-500'
                            }`} />
                            <div className="flex-1">
                              <h4 className="text-sm font-medium text-gray-900 thai-text">
                                {notification.title}
                              </h4>
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
                      <button className="w-full text-center text-sm text-primary-600 hover:text-primary-700 font-medium thai-text">
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
                onClick={() => setShowUserMenu(!showUserMenu)}
                className={`
                  flex items-center space-x-3 p-1.5 pr-4 rounded-2xl transition-all duration-300
                  ${showUserMenu ? 'bg-indigo-50 shadow-sm' : 'hover:bg-slate-50'}
                `}
              >
                <div className="w-9 h-9 bg-gradient-to-br from-indigo-500 to-indigo-600 rounded-xl flex items-center justify-center shadow-md shadow-indigo-200">
                  <UserIcon className="h-5 w-5 text-white" />
                </div>
                <div className="hidden md:block text-left">
                  <p className="text-sm font-bold text-slate-900 thai-text leading-tight">
                    ผู้ดูแลระบบ
                  </p>
                  <p className="text-[10px] font-bold text-indigo-500 uppercase tracking-tighter">
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
                      <p className="text-sm font-medium text-gray-900 thai-text">
                        ผู้ดูแลระบบ
                      </p>
                      <p className="text-xs text-gray-500">
                        admin@company.com
                      </p>
                    </div>
                    <div className="py-2">
                      <button className="flex items-center w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors duration-150">
                        <SettingsIcon className="h-4 w-4 mr-3 text-gray-400" />
                        <span className="thai-text">ตั้งค่าบัญชี</span>
                      </button>
                      <button className="flex items-center w-full px-4 py-2 text-sm text-error-600 hover:bg-error-50 transition-colors duration-150">
                        <LogOutIcon className="h-4 w-4 mr-3 text-error-500" />
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
          onClick={() => {
            setShowUserMenu(false)
            setShowNotifications(false)
          }}
        />
      )}
    </header>
  )
}
