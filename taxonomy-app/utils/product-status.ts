/**
 * สถานะสินค้าที่ pipeline เขียนจริง (`app/api/import/commit/route.ts` และหน้า Data Quality)
 * ไม่มี `'pending'` — UI ที่เคยกรองค่านั้นจึงว่างเปล่าถาวร
 *
 * แยกออกจาก `utils/supabase.ts` เพราะไฟล์นั้นสร้าง Supabase client ตอน import
 * โค้ดฝั่ง server และเทสต์จึงอ้างค่าคงที่พวกนี้ได้โดยไม่ต้องมี env ของ client
 */
export type ProductStatus =
  | 'pending_review_dedup'
  | 'pending_review_category'
  | 'approved'
  | 'rejected'
  | 'draft'

/** สองด่านที่ยังรอคนตรวจ ใช้ร่วมกันระหว่างหน้าแรกกับหน้ารายการสินค้า */
export const PENDING_REVIEW_STATUSES: ProductStatus[] = [
  'pending_review_dedup',
  'pending_review_category'
]
