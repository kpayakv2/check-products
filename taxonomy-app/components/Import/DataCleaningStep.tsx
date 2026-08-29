'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { SparklesIcon, CheckCircleIcon, ArrowRightIcon, BrainIcon } from 'lucide-react'
import { ParsedCSV } from '@/utils/csv-parser'
import { ColumnMapping } from './ColumnMappingStep'
import { parsePrice } from '@/utils/price'
import type { WizardItem } from '@/types/import'

interface DataCleaningStepProps {
  parsedData: ParsedCSV
  columnMapping: ColumnMapping
  onComplete: (cleanedData: WizardItem[]) => void
  onBack: () => void
}

export default function DataCleaningStep({
  parsedData,
  columnMapping,
  onComplete,
  onBack
}: DataCleaningStepProps) {
  const [isCleaning, setIsCleaning] = useState(true)
  const [cleanedResults, setCleanedResults] = useState<WizardItem[]>([])
  const [stats, setStats] = useState({ total: 0, changed: 0 })

  useEffect(() => {
    cleanData()
  }, [])

  const cleanData = async () => {
    setIsCleaning(true)
    try {
      const productNameCol = columnMapping.product_name
      const rawTexts = parsedData.rows.map(row => String(row[productNameCol] || ''))

      // Use absolute path for API
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'
      const response = await fetch(`${apiUrl}/api/v1/clean`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts: rawTexts })
      })

      if (!response.ok) throw new Error('Failed to clean data')
      
      const data = await response.json()
      
      const priceCol = columnMapping.price
      const mergedResults = parsedData.rows.map((row, i) => {
        const cleanedText = data.results[i]?.cleaned || row[productNameCol]
        return {
          ...row,
          _original_name: row[productNameCol],
          _cleaned_name: cleanedText,
          _is_changed: row[productNameCol] !== cleanedText,
          price: priceCol ? parsePrice(row[priceCol]) : undefined
        }
      })

      setCleanedResults(mergedResults)
      setStats({
        total: mergedResults.length,
        changed: mergedResults.filter(r => r._is_changed).length
      })
    } catch (error) {
      console.error("Cleaning error:", error)
      // Fallback
      const priceCol = columnMapping.price
      const mergedResults = parsedData.rows.map(row => ({
        ...row,
        _original_name: row[columnMapping.product_name],
        _cleaned_name: row[columnMapping.product_name],
        _is_changed: false,
        price: priceCol ? parsePrice(row[priceCol]) : undefined
      }))
      setCleanedResults(mergedResults)
      setStats({ total: mergedResults.length, changed: 0 })
    } finally {
      setIsCleaning(false)
    }
  }

  if (isCleaning) {
    return (
      <div className="flex flex-col items-center justify-center py-32 space-y-6">
        <div className="relative">
          <div className="w-24 h-24 border-4 border-indigo-100 border-t-indigo-600 rounded-full animate-spin"></div>
          <SparklesIcon className="w-8 h-8 text-indigo-600 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-pulse" />
        </div>
        <div className="text-center">
          <h3 className="text-xl font-bold text-slate-800 thai-text">กำลังทำความสะอาดข้อมูล...</h3>
          <p className="text-slate-500 thai-text">ลบคำฟุ่มเฟือย แปลงเลขไทย และจัดรูปแบบตัวอักษร</p>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-6xl mx-auto space-y-8 pb-20">
      {/* Header Info */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-black text-slate-900 tracking-tight font-noto-sans-thai mb-2 flex items-center gap-3">
            <SparklesIcon className="w-8 h-8 text-indigo-600" />
            ข้อมูลที่ทำความสะอาดแล้ว
          </h2>
          <p className="text-slate-500 font-medium thai-text">ตรวจสอบผลลัพธ์การลบคำขยะและสระลอยก่อนนำไปหาของซ้ำ</p>
        </div>
        <div className="flex gap-4">
          <div className="bg-indigo-50 px-6 py-3 rounded-2xl text-center">
            <div className="text-2xl font-black text-indigo-600">{stats.total}</div>
            <div className="text-xs font-black text-indigo-900 uppercase tracking-widest thai-text">รายการทั้งหมด</div>
          </div>
          <div className="bg-emerald-50 px-6 py-3 rounded-2xl text-center">
            <div className="text-2xl font-black text-emerald-600">{stats.changed}</div>
            <div className="text-xs font-black text-emerald-900 uppercase tracking-widest thai-text">ถูกแก้ไข</div>
          </div>
        </div>
      </div>

      {/* Preview Table */}
      <div className="premium-card p-0 overflow-hidden bg-white border border-slate-200 shadow-xl rounded-3xl">
        <div className="overflow-x-auto max-h-[500px]">
          <table className="w-full text-left border-collapse">
            <thead className="sticky top-0 bg-slate-50/90 backdrop-blur-md shadow-sm z-10">
              <tr>
                <th className="px-6 py-4 text-xs font-black text-slate-500 uppercase tracking-wider thai-text w-1/2">
                  ชื่อดั้งเดิม (Original)
                </th>
                <th className="px-6 py-4 text-xs font-black text-indigo-600 uppercase tracking-wider thai-text w-1/2">
                  ชื่อที่ทำความสะอาดแล้ว (Cleaned)
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {cleanedResults.map((row, i) => (
                <tr key={i} className="hover:bg-slate-50/50 transition-colors">
                  <td className="px-6 py-4">
                    <span className="text-sm font-medium text-slate-500 line-clamp-2">
                      {row._original_name}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex items-start gap-2">
                      <span className={`text-sm font-bold line-clamp-2 ${row._is_changed ? 'text-indigo-700 bg-indigo-50 px-2 py-0.5 rounded' : 'text-slate-700'}`}>
                        {row._cleaned_name}
                      </span>
                      {row._is_changed && (
                        <span className="shrink-0 inline-flex items-center gap-1 text-[10px] font-bold text-emerald-600 bg-emerald-50 px-2 py-1 rounded-full uppercase tracking-wider">
                          <CheckCircleIcon className="w-3 h-3" /> แก้ไขแล้ว
                        </span>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Footer Actions */}
      <div className="flex items-center justify-between pt-8">
        <button
          onClick={onBack}
          className="px-8 py-4 text-slate-500 hover:text-slate-900 font-bold text-sm transition-all thai-text"
        >
          ← ย้อนกลับ
        </button>

        <button
          onClick={() => onComplete(cleanedResults)}
          className="px-10 py-4 bg-slate-900 hover:bg-black text-white rounded-[2rem] font-black text-sm tracking-wide shadow-xl shadow-slate-900/20 transition-all active:scale-95 flex items-center gap-3 thai-text"
        >
          นำไปจัดการของซ้ำ
          <ArrowRightIcon className="w-5 h-5" />
        </button>
      </div>
    </div>
  )
}
