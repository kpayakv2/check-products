/**
 * ออกรหัสสินค้า (Item ID) ให้สินค้าใหม่ที่ยังไม่มี
 *
 * POS บังคับช่องนี้ แต่สินค้าที่มาจาก import wizard ไม่มี sku เลย และไฟล์ต้นทาง
 * ก็ไม่มีคอลัมน์รหัส จึงต้องออกเอง — โดยเข้าชุดกับสตอกเดิม 3,103 รายการ
 * ที่ใช้รูปแบบ <PREFIX><เลขรัน> และ prefix ผูกกับหมวดหลัก
 *
 * สเปกเต็ม: docs/specs/2026-09-06-pos-export.md
 */

/**
 * หมวดหลัก → prefix อ่านออกมาจากสตอกเดิมทั้ง 3,103 รายการ
 *
 * เป็นตารางคงที่ ไม่ derive จากฐานข้อมูลตอนรัน เพราะหมวดที่ยังไม่เคยมีสินค้าเก่า
 * จะหา prefix ไม่เจอ และการเปลี่ยน prefix โดยไม่ตั้งใจแปลว่ารหัสที่ส่งเข้า POS
 * ไปแล้วกับที่จะส่งรอบหน้าอยู่คนละชุดกัน
 */
export const MAIN_CATEGORY_PREFIX: Record<string, string> = {
  'ผลิตภัณฑ์ดูแลส่วนบุคคล': 'PC',
  'ของใช้ในบ้าน': 'HH',
  'เครื่องครัว': 'KIT',
  'เครื่องมือ_ฮาร์ดแวร์': 'HW',
  'เบ็ดเตล็ด': 'MIS',
  'ผลิตภัณฑ์ทำความสะอาดในบ้าน': 'CLH',
  'เครื่องเขียน_สำนักงาน': 'ST',
  'อุปกรณ์ทำความสะอาด': 'CL',
  'ของเล่น_นันทนาการ': 'TOY',
  'สินค้าเพื่อสัตว์เลี้ยง': 'PET',
  'เครื่องใช้ไฟฟ้า': 'ELC',
  'แม่และเด็ก': 'BB',
  'อาหาร_เครื่องดื่ม': 'FD',
  'ของใช้ในพิธีกรรม': 'CER',
  'เวชภัณฑ์_สุขภาพ': 'MED',
  'ยานยนต์': 'AUT',
}

/** หมวดที่ไม่อยู่ในตารางตกมาที่นี่ — MIS คือหมวดเบ็ดเตล็ดของสตอกเดิมอยู่แล้ว */
export const FALLBACK_PREFIX = 'MIS'

const SKU_PATTERN = /^([A-Za-z]+)(\d+)$/

export interface ItemIdCandidate {
  id: string
  name_th: string
  sku: string | null
  mainCategory: string | null
}

export function prefixForMainCategory(mainCategory: string | null | undefined): string {
  const key = (mainCategory ?? '').trim()
  if (!key) return FALLBACK_PREFIX
  return MAIN_CATEGORY_PREFIX[key] ?? FALLBACK_PREFIX
}

/** เลขรันสูงสุดที่ใช้ไปแล้วของแต่ละ prefix — รหัสที่ไม่เข้ารูปแบบถูกข้าม ไม่ทำให้พังทั้งชุด */
export function maxNumberByPrefix(existingSkus: readonly string[]): Map<string, number> {
  const maxes = new Map<string, number>()

  for (const raw of existingSkus) {
    const match = SKU_PATTERN.exec((raw ?? '').trim())
    if (!match) continue

    const prefix = match[1].toUpperCase()
    const num = Number(match[2])
    if (!Number.isFinite(num)) continue

    maxes.set(prefix, Math.max(maxes.get(prefix) ?? 0, num))
  }

  return maxes
}

/**
 * แจกรหัสให้เฉพาะรายการที่ยังไม่มี sku
 *
 * เรียงตามชื่อก่อนแจกเสมอ เพื่อให้รันด้วยข้อมูลชุดเดิมได้ผลเดิมทุกครั้ง —
 * หน้าตัวอย่าง (GET) กับตอนดาวน์โหลดจริง (POST) ต้องแสดงรหัสตรงกัน
 * ไม่งั้นคนเห็นรหัสหนึ่งบนจอ แต่ไฟล์ที่ได้เป็นอีกรหัส
 *
 * `taken` กันชนกับรหัสที่มีอยู่แล้ว เผื่อเลขในฐานข้อมูลขาดช่วง —
 * products.sku มี UNIQUE constraint การ insert ที่ชนจะล้มทั้งชุด
 */
export function allocateItemIds(
  candidates: readonly ItemIdCandidate[],
  existingSkus: readonly string[]
): Map<string, string> {
  const nextByPrefix = maxNumberByPrefix(existingSkus)
  const taken = new Set(existingSkus.map((sku) => (sku ?? '').trim().toUpperCase()))
  const assigned = new Map<string, string>()

  const pending = candidates
    .filter((item) => !(item.sku ?? '').trim())
    .slice()
    .sort((a, b) => a.name_th.localeCompare(b.name_th, 'th'))

  for (const item of pending) {
    const prefix = prefixForMainCategory(item.mainCategory)
    let next = (nextByPrefix.get(prefix) ?? 0) + 1

    while (taken.has(`${prefix}${next}`)) next += 1

    nextByPrefix.set(prefix, next)
    const sku = `${prefix}${next}`
    taken.add(sku)
    assigned.set(item.id, sku)
  }

  return assigned
}
