'use client'

import { useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { toast } from 'react-hot-toast'
import { LockIcon } from 'lucide-react'

import { FOCUS_RING } from '@/components/ui/Modal'

/**
 * next มาจาก query string ที่ผู้ใช้พิมพ์เอง — ต้องเป็น path ภายในแอปเท่านั้น
 * ('/xxx' ที่ไม่ใช่ '//xxx' ซึ่งเบราว์เซอร์ตีความเป็น protocol-relative URL ออกนอกโดเมนได้)
 * ไม่งั้นเป็นช่องโหว่ open redirect: ลิงก์ /unlock?next=https://evil.com หลอกให้กรอกรหัส
 * แล้วพาออกนอกแอปไปเว็บปลอมได้
 */
function safeNextPath(next: string | null): string {
  if (!next || !next.startsWith('/') || next.startsWith('//')) return '/'
  return next
}

export default function UnlockPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
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
      router.push(safeNextPath(searchParams.get('next')))
      router.refresh()
    } catch {
      toast.error('เกิดข้อผิดพลาด กรุณาลองใหม่')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-4 bg-slate-50">
      <main className="w-full max-w-sm bg-white rounded-2xl shadow-lg border border-slate-200 p-8">
        <div className="flex flex-col items-center mb-6">
          <div className="bg-indigo-600 rounded-full p-3 mb-3 shadow-md">
            <LockIcon className="w-6 h-6 text-white" aria-hidden="true" />
          </div>
          <h1 className="text-xl font-bold text-slate-900 thai-text">เข้าสู่ระบบ</h1>
          <p className="text-sm text-slate-600 mt-1 text-center thai-text">
            กรอกรหัสผ่านสำหรับใช้งานภายในองค์กร
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="secret-input" className="sr-only">
              รหัสผ่าน
            </label>
            <input
              id="secret-input"
              type="password"
              value={secret}
              onChange={(e) => setSecret(e.target.value)}
              placeholder="รหัสผ่าน"
              autoComplete="current-password"
              autoFocus
              className={`w-full px-4 py-2.5 border border-slate-300 rounded-xl text-slate-900 placeholder:text-slate-400 ${FOCUS_RING}`}
            />
          </div>
          <button
            type="submit"
            disabled={submitting || !secret}
            className={`w-full bg-indigo-600 text-white py-2.5 rounded-xl font-bold text-sm hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors ${FOCUS_RING}`}
          >
            {submitting ? 'กำลังตรวจสอบ...' : 'ปลดล็อก'}
          </button>
        </form>
      </main>
    </div>
  )
}
