'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import ApprovalStep from '@/components/Import/ApprovalStep'
import Sidebar from '@/components/Layout/Sidebar'
import Header from '@/components/Layout/Header'
import { 
  Clock,
  ArrowLeft,
  CheckCircle
} from 'lucide-react'

export default function PendingApprovalsPage() {
  const [completedCount, setCompletedCount] = useState(0)

  const handleApprovalComplete = (results: any) => {
    setCompletedCount(prev => prev + results.results.success)
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Sidebar />
      <div className="lg:pl-64">
        <Header />
        
        <main className="p-6">
          <div className="max-w-7xl mx-auto">
            {/* Page Header */}
            <div className="mb-8">
              <div className="flex items-center space-x-3 mb-4">
                <div className="p-2 bg-orange-100 rounded-lg">
                  <Clock className="w-6 h-6 text-orange-600" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-gray-900">
                    รายการรอการอนุมัติ
                  </h1>
                  <p className="text-gray-600">
                    ตรวจสอบและอนุมัติสินค้าที่ AI ประมวลผลแล้ว
                  </p>
                </div>
              </div>

              {/* Navigation */}
              <div className="flex items-center justify-between">
                <nav className="flex items-center space-x-4 text-sm">
                  <a 
                    href="/import" 
                    className="text-gray-500 hover:text-gray-700 transition-colors flex items-center space-x-1"
                  >
                    <ArrowLeft className="w-4 h-4" />
                    <span>กลับไปหน้า Import</span>
                  </a>
                  <span className="text-gray-300">/</span>
                  <span className="text-gray-900 font-medium">รายการรอการอนุมัติ</span>
                </nav>

                {completedCount > 0 && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="flex items-center space-x-2 bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm"
                  >
                    <CheckCircle className="w-4 h-4" />
                    <span>อนุมัติแล้ว {completedCount} รายการ</span>
                  </motion.div>
                )}
              </div>
            </div>

            {/* Approval Interface */}
            <div className="bg-white rounded-lg shadow-sm border">
              <ApprovalStep 
                onComplete={handleApprovalComplete}
                onBack={() => window.history.back()}
              />
            </div>

            {/* Help Text */}
            <div className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h3 className="font-medium text-blue-900 mb-2">💡 วิธีใช้งาน</h3>
              <ul className="text-sm text-blue-800 space-y-1">
                <li>• คลิกเพื่อขยายดูรายละเอียดของแต่ละสินค้า</li>
                <li>• ตรวจสอบหมวดหมู่ที่ AI แนะนำและคุณสมบัติที่สกัดได้</li>
                <li>• เลือกหลายรายการเพื่ออนุมัติ/ปฏิเสธแบบ batch</li>
                <li>• สินค้าที่อนุมัติแล้วจะถูกบันทึกลงฐานข้อมูลทันที</li>
                <li>• คุณสามารถกลับมาดำเนินการต่อได้ทุกเมื่อ</li>
              </ul>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
