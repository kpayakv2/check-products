/**
 * แปลง/เทียบราคาสินค้า ใช้ทั้งตอนอ่านค่าจากไฟล์นำเข้าและตอนตรวจของซ้ำ
 */

export function parsePrice(raw: string | number | undefined | null): number | undefined {
  if (raw === undefined || raw === null || raw === '') return undefined
  const cleaned = String(raw).replace(/[,฿\s]/g, '')
  const num = parseFloat(cleaned)
  return Number.isFinite(num) && num > 0 ? num : undefined
}

/**
 * ราคาใดราคาหนึ่งไม่มีข้อมูล ให้ถือว่าไม่ขัดแย้งกัน (ข้ามการเช็ค ไม่นับเป็นจุดลบ)
 * เพราะราคาฝั่งสต๊อกส่วนใหญ่ยังไม่ถูก backfill
 */
export function isPriceMismatch(
  newPrice: number | undefined | null,
  oldPrice: number | undefined | null,
  minRatio = 0.5
): boolean {
  if (!newPrice || !oldPrice) return false
  const ratio = Math.min(newPrice, oldPrice) / Math.max(newPrice, oldPrice)
  return ratio <= minRatio
}

export type DedupBucket = 'duplicate' | 'review' | 'new'

/**
 * ตัดสิน bucket ของสินค้าที่พบ candidate จาก embedding matching
 * ราคาไม่เคยบล็อกการจับคู่แบบเด็ดขาด — ใช้ลดชั้นความมั่นใจเฉพาะโซน auto-merge (≥95%)
 * เท่านั้น โซน 80-94% ต้องมีคนตรวจอยู่แล้วไม่ว่าราคาจะขัดแย้งกันหรือไม่ จึง bucket เดิม
 */
export function classifyDedupBucket(
  score: number,
  mlPrediction: string | undefined,
  newPrice: number | undefined | null,
  oldPrice: number | undefined | null
): DedupBucket {
  if (score >= 0.95) {
    return isPriceMismatch(newPrice, oldPrice) ? 'review' : 'duplicate'
  }
  if (mlPrediction === 'similar' || score >= 0.8) {
    return 'review'
  }
  return 'new'
}
