'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import toast from 'react-hot-toast'
import { LockIcon } from 'lucide-react'

export default function UnlockPage() {
  const router = useRouter()
  const [secret, setSecret] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSubmitting(true)

    try {
      const res = await fetch('/api/unlock', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ secret }),
      })
      const data = await res.json()

      if (!res.ok || !data.success) {
        toast.error(data.error || 'รหัสผ่านไม่ถูกต้อง')
        return
      }

      toast.success('ปลดล็อกสำเร็จ')
      router.push('/')
      router.refresh()
    } catch {
      toast.error('เกิดข้อผิดพลาด กรุณาลองใหม่')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <div className="w-full max-w-sm bg-white rounded-2xl shadow-lg border border-gray-200 p-8">
        <div className="flex flex-col items-center mb-6">
          <div className="bg-indigo-600 rounded-full p-3 mb-3">
            <LockIcon className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-lg font-semibold text-gray-900">เข้าสู่ระบบ</h1>
          <p className="text-sm text-gray-500 mt-1 text-center">
            กรอกรหัสผ่านสำหรับใช้งานภายในองค์กร
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <input
            type="password"
            value={secret}
            onChange={(e) => setSecret(e.target.value)}
            placeholder="รหัสผ่าน"
            autoFocus
            className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
          />
          <button
            type="submit"
            disabled={submitting || !secret}
            className="w-full bg-indigo-600 text-white py-2.5 rounded-lg font-medium hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {submitting ? 'กำลังตรวจสอบ...' : 'ปลดล็อก'}
          </button>
        </form>
      </div>
    </div>
  )
}
