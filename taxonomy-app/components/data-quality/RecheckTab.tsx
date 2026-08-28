'use client'

import { useCallback, useEffect, useState } from 'react'
import { AlertCircle, ArrowRight, Check, RefreshCcw, User, Bot } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { supabase } from '@/utils/supabase'

interface RecheckItem {
  id: string
  confidence_score: number
  suggested_category_id: string | null
  product: { id: string; name_th: string; sku: string | null; category_id: string | null }
  suggested_category: { id: string; name_th: string } | null
  metadata: {
    current_category?: string
    current_category_id?: string
    suggested_category?: string
    alternatives?: { category: string; confidence: number }[]
  }
}

interface TaxonomyNode {
  id: string
  name_th: string
}

const PAGE_SIZE = 25

/**
 * รายการที่ AI ตรวจซ้ำแล้วเห็นต่างจากหมวดที่คนจัดไว้
 *
 * แสดงหมวดเดิมคู่กับหมวดที่ AI เสนอ เพื่อให้คนตัดสินได้เร็ว
 * ทุกครั้งที่ยืนยัน ระบบจะเรียนคีย์เวิร์ดจากหมวดนั้นต่อ (ผ่าน /api/recheck)
 */
export default function RecheckTab() {
  const [items, setItems] = useState<RecheckItem[]>([])
  const [taxonomy, setTaxonomy] = useState<TaxonomyNode[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [busyId, setBusyId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const fetchItems = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`/api/recheck?limit=${PAGE_SIZE}`)
      const payload = await response.json()
      if (!payload.success) throw new Error(payload.error || 'โหลดข้อมูลไม่สำเร็จ')
      setItems(payload.data)
      setTotal(payload.pagination.total)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'โหลดข้อมูลไม่สำเร็จ')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchItems()
  }, [fetchItems])

  // รายชื่อหมวดสำหรับ dropdown "เลือกหมวดอื่น" — โหลดครั้งเดียว
  useEffect(() => {
    supabase
      .from('taxonomy_nodes')
      .select('id, name_th')
      .eq('level', 1)
      .order('name_th')
      .then(({ data }) => setTaxonomy(data ?? []))
  }, [])

  const decide = async (
    suggestionId: string,
    action: 'keep' | 'accept' | 'override',
    categoryId?: string
  ) => {
    setBusyId(suggestionId)
    setError(null)
    try {
      const response = await fetch('/api/recheck', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ suggestion_id: suggestionId, action, category_id: categoryId }),
      })
      const payload = await response.json()
      if (!payload.success) throw new Error(payload.error || 'บันทึกไม่สำเร็จ')

      // เอารายการที่ตัดสินแล้วออกทันที ไม่ต้องรอโหลดใหม่ทั้งหน้า
      setItems((current) => current.filter((item) => item.id !== suggestionId))
      setTotal((current) => Math.max(current - 1, 0))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'บันทึกไม่สำเร็จ')
    } finally {
      setBusyId(null)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-16 text-slate-500" data-testid="recheck-loading">
        <RefreshCcw className="w-5 h-5 animate-spin mr-2" />
        กำลังโหลดรายการที่ต้องตรวจ...
      </div>
    )
  }

  return (
    <div data-testid="recheck-tab">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-5">
        <div>
          <h3 className="text-lg font-black text-slate-800">ตรวจซ้ำหมวดหมู่สินค้าเก่า</h3>
          <p className="text-sm text-slate-500">
            แสดงเฉพาะรายการที่ AI เห็นต่างจากหมวดที่คนจัดไว้ —{' '}
            <span className="font-bold text-slate-700" data-testid="recheck-total">
              เหลือ {total.toLocaleString()} รายการ
            </span>
          </p>
        </div>
        <button
          onClick={fetchItems}
          className="flex items-center gap-2 px-3 py-2 text-sm font-bold text-slate-600 bg-slate-100 rounded-lg hover:bg-slate-200"
          data-testid="recheck-refresh"
        >
          <RefreshCcw className="w-4 h-4" />
          โหลดใหม่
        </button>
      </div>

      {error && (
        <div className="flex items-center gap-2 p-3 mb-4 text-sm font-bold text-rose-700 bg-rose-50 border border-rose-200 rounded-lg">
          <AlertCircle className="w-4 h-4 shrink-0" />
          {error}
        </div>
      )}

      {items.length === 0 ? (
        <div className="py-16 text-center text-slate-500" data-testid="recheck-empty">
          <Check className="w-10 h-10 mx-auto mb-3 text-emerald-500" />
          <p className="font-bold">ตรวจครบแล้ว ไม่มีรายการค้าง</p>
        </div>
      ) : (
        <div className="space-y-3">
          <AnimatePresence initial={false}>
            {items.map((item) => (
              <motion.div
                key={item.id}
                layout
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, height: 0, marginBottom: 0 }}
                className="p-4 bg-white border border-slate-200 rounded-xl"
                data-testid={`recheck-row-${item.id}`}
              >
                <div className="flex flex-wrap items-start justify-between gap-2 mb-3">
                  <p className="font-bold text-slate-800 break-words min-w-0">
                    {item.product.name_th}
                  </p>
                  <span className="shrink-0 px-2 py-1 text-xs font-black rounded-md bg-slate-100 text-slate-600">
                    มั่นใจ {(item.confidence_score * 100).toFixed(0)}%
                  </span>
                </div>

                {/* หมวดเดิมของคน คู่กับหมวดที่ AI เสนอ */}
                <div className="flex flex-wrap items-center gap-2 mb-4 text-sm">
                  <span className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-slate-100 text-slate-700 font-bold">
                    <User className="w-3.5 h-3.5 shrink-0" />
                    <span className="break-words">{item.metadata.current_category ?? '—'}</span>
                  </span>
                  <ArrowRight className="w-4 h-4 text-slate-400 shrink-0" />
                  <span className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-indigo-50 text-indigo-700 font-bold">
                    <Bot className="w-3.5 h-3.5 shrink-0" />
                    <span className="break-words">{item.suggested_category?.name_th ?? '—'}</span>
                  </span>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  <button
                    disabled={busyId === item.id}
                    onClick={() => decide(item.id, 'keep')}
                    className="px-3 py-2 text-sm font-bold rounded-lg bg-slate-700 text-white hover:bg-slate-800 disabled:opacity-50"
                    data-testid={`recheck-keep-${item.id}`}
                  >
                    คงหมวดเดิม
                  </button>
                  <button
                    disabled={busyId === item.id || !item.suggested_category_id}
                    onClick={() => decide(item.id, 'accept')}
                    className="px-3 py-2 text-sm font-bold rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-50"
                    data-testid={`recheck-accept-${item.id}`}
                  >
                    ใช้ของ AI
                  </button>
                  <select
                    disabled={busyId === item.id}
                    defaultValue=""
                    onChange={(event) => {
                      if (event.target.value) decide(item.id, 'override', event.target.value)
                    }}
                    className="px-3 py-2 text-sm font-bold rounded-lg border border-slate-300 bg-white max-w-full"
                    data-testid={`recheck-override-${item.id}`}
                  >
                    <option value="">เลือกหมวดอื่น...</option>
                    {taxonomy.map((node) => (
                      <option key={node.id} value={node.id}>
                        {node.name_th}
                      </option>
                    ))}
                  </select>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      )}
    </div>
  )
}
