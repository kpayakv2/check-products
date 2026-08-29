/**
 * ชนิดข้อมูลกลางของ Import Wizard
 *
 * สินค้าหนึ่งรายการถูก spread ต่อกันไปเรื่อย ๆ ทีละขั้น (ล้างข้อมูล → ตรวจของซ้ำ → จัดหมวดหมู่)
 * แต่ละขั้นเติมคีย์ที่ขึ้นต้นด้วย _ เข้าไป ก่อนหน้านี้ทุก step รับ props เป็น any ทั้งก้อน
 * จึงไม่มีอะไรจับได้เลยเวลาชื่อคีย์ไม่ตรงกันระหว่างขั้น ซึ่งเป็นต้นเหตุของบั๊กหลายรอบที่ผ่านมา
 */

// ผลตัดสินว่าซ้ำหรือไม่ นิยามอยู่คู่กับ classifyDedupBucket ที่เป็นคนกำหนดค่าอยู่แล้ว
// ดึงมาใช้ต่อ ไม่ประกาศซ้ำ จะได้ไม่มีวันหลุดจากกัน
export type { DedupBucket } from '@/utils/price'
import type { DedupBucket } from '@/utils/price'

export interface WizardItem {
  /** คอลัมน์ดิบจาก CSV ที่ติดมาตั้งแต่ต้น ชื่อคอลัมน์ขึ้นกับไฟล์ที่ผู้ใช้อัปโหลด */
  [column: string]: unknown

  // ขั้นล้างข้อมูล
  _original_name?: string
  _cleaned_name?: string
  _is_changed?: boolean
  price?: number

  // ขั้นตรวจของซ้ำ
  _similarity_score?: number
  _matched_with?: string
  /** id จริงของสินค้าในสตอก ใช้เป็น FK ได้ ไม่ใช่เลขลำดับรีวิว */
  _matched_id?: string
  _new_price?: number
  _old_price?: number
  _price_mismatch?: boolean
  _bucket?: DedupBucket
  /** true เมื่อคนกดตัดสินเอง ไม่ใช่ค่าที่ระบบเดาให้ */
  _reviewed_by_user?: boolean
  /** 'mock' คือผลจำลองตอน backend ล่ม ห้ามบันทึกลงฐานข้อมูลเด็ดขาด */
  _source?: 'backend' | 'mock'

  // ขั้นจัดหมวดหมู่
  _suggested_category?: string
  _suggested_category_id?: string
  _confidence?: number
}

export interface DedupResults {
  autoMerged: WizardItem[]
  autoCreated: WizardItem[]
  reviewZone: WizardItem[]
}

export interface SavedProduct {
  id: string
  name_th: string
  status: string
}

/** ผลตอบกลับจริงจาก /api/import/commit — หน้าสรุปต้องอ่านตัวเลขจากตรงนี้เท่านั้น */
export interface SaveResult {
  success: boolean
  import_batch_id: string
  saved: number
  counts: Record<string, number>
  products: SavedProduct[]
  similarity_pairs?: number
  embedded?: number
  missing_embedding?: number
  /** true เมื่อรอบนี้เคยบันทึกไปแล้ว เซิร์ฟเวอร์คืนผลเดิมแทนการบันทึกซ้ำ */
  reused?: boolean
  categorized?: number
}
