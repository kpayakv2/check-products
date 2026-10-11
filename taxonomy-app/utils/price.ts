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

/** คำตัดสินของกฎอ่านตัวเลขจาก backend (src/core/product_identity.py) — กฎอยู่ที่เดียวเท่านั้น */
export type IdentityVerdict = 'different' | 'duplicate' | 'needs_judgment'

/**
 * ตัดสิน bucket ของสินค้าที่พบ candidate จาก embedding matching
 *
 * 'duplicate' คือการทิ้งสินค้านำเข้าโดยไม่มีคนเห็นเลย (ต่างจาก 'new' ที่ยังมีคนยืนยันหมวดต่อ)
 * จึงต้องเข้มกว่า: คะแนน ≥95% อย่างเดียวไม่พอ ต้องให้กฎอ่านตัวเลขยืนยันว่าซ้ำด้วย
 * วัดกับไฟล์ทดสอบ 405 รายการแล้วคะแนนอย่างเดียวทิ้งผิด 3 จาก 37 (`กะละมัง 004 58` ↔ `004/53`)
 * ถ้า backend ไม่ส่งคำตัดสินมา (undefined) ถือว่ายังไม่ยืนยัน ส่งให้คนดู ไม่ทิ้งเอง
 *
 * กฎว่า "คนละสินค้า" → 'new' ทุกช่วงคะแนน ราคาไม่เคยบล็อกการจับคู่แบบเด็ดขาด —
 * ใช้ลดชั้นความมั่นใจเฉพาะโซนทิ้ง (≥95%) เท่านั้น
 */
export function classifyDedupBucket(
  score: number,
  mlPrediction: string | undefined,
  newPrice: number | undefined | null,
  oldPrice: number | undefined | null,
  identity?: IdentityVerdict
): DedupBucket {
  if (identity === 'different') {
    return 'new'
  }
  if (score >= 0.95) {
    return identity === 'duplicate' && !isPriceMismatch(newPrice, oldPrice) ? 'duplicate' : 'review'
  }
  if (mlPrediction === 'similar' || score >= 0.8) {
    return 'review'
  }
  return 'new'
}
