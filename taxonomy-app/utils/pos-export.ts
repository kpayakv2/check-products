/**
 * แปลงสินค้า + taxonomy เป็นไฟล์ .txt ที่ POS นำเข้าได้
 *
 * ฟังก์ชันในไฟล์นี้บริสุทธิ์ทั้งหมด (ไม่แตะฐานข้อมูล) เพราะเงื่อนไขของไฟล์
 * ละเอียดกว่าที่ตาเห็นมาก — ไม้เอกซ้ำในหัวคอลัมน์, CRLF, ไม่มี BOM, ไม่มี
 * เครื่องหมายคำพูด ทดสอบตรงนี้ให้ครบดีกว่าไปเจอตอน POS ปฏิเสธไฟล์
 *
 * สเปกเต็ม: docs/specs/2026-09-06-pos-export.md
 */

/**
 * หัวตาราง 9 คอลัมน์ ต้องตรงตัวอักษรกับไฟล์ที่ POS เคยรับ
 *
 * คอลัมน์ที่ 5 สะกดผิดในต้นฉบับ — ไม้เอก (U+0E48) ซ้ำสองตัวต่อจากสระอือ
 * จึงเขียนด้วย escape ไม่ใช่ตัวอักษรตรง ๆ เพราะไม้เอกซ้อนกันมองด้วยตาแทบไม่เห็น
 * เผลอโดน formatter หรือ IDE "แก้คำผิด" ให้เมื่อไหร่ ไฟล์จะใช้กับ POS ไม่ได้
 * และจะไม่มี error ให้จับ — ห้ามแก้
 */
export const POS_EXPORT_HEADERS = [
  'Item Group(ชื่อกลุ่มสินค้า)',
  'Sub Group1(ชื่อกลุ่มย่อย1)',
  'Item ID(รหัสสินค้า)',
  'Barcode(บาร์โค้ด)',
  'Item Name(\u0E0A\u0E37\u0E48\u0E48\u0E2D\u0E2A\u0E34\u0E19\u0E04\u0E49\u0E32)',
  'Unit Name(ชื่อหน่วย)',
  'Price(ราคาขาย)',
  'Vat',
  'Sub Group2',
] as const

const FIELD_SEPARATOR = '\t'
const ROW_SEPARATOR = '\r\n'
const DEFAULT_VAT = 'V'

/**
 * POS บังคับช่องหน่วยนับ แต่ระบบไม่เคยเก็บหน่วยขายเลย และไฟล์ต้นทางก็ไม่มี
 * เติมตรงนี้ตอนประกอบไฟล์ ไม่ใช่ตั้ง default ในฐานข้อมูล — ฐานข้อมูลไม่ควรอ้างว่า
 * รู้หน่วยของสินค้าเก่า 3,103 รายการทั้งที่ไม่เคยมีใครกรอก
 */
export const DEFAULT_UNIT = 'ชิ้น'

export interface TaxonomyNodeLite {
  id: string
  name_th: string
  parent_id: string | null
}

export interface ExportableProduct {
  name_th: string
  sku: string | null
  barcode: string | null
  unit_name: string | null
  vat_flag: string | null
  price: number | null
  category_id: string | null
}

export interface PosExportRow {
  itemGroup: string
  subGroup1: string
  subGroup2: string
  itemId: string
  barcode: string
  itemName: string
  unitName: string
  price: number | null
  vat: string
}

/** หนึ่งช่องต่อหนึ่งช่องบังคับของ POS — ขาดช่องไหน POS ไม่รับไฟล์ */
export interface PosExportIssues {
  missingItemId: number
  missingBarcode: number
  missingItemName: number
  missingUnitName: number
  missingItemGroup: number
  missingPrice: number
}

/**
 * ล้างอักขระที่จะทำให้โครงไฟล์พัง
 * ไฟล์ไม่มีเครื่องหมายคำพูดครอบ tab หรือ newline ที่หลุดมาในค่าเดียว
 * จึงกลายเป็นคอลัมน์ใหม่หรือแถวใหม่ทันที
 */
function sanitizeField(value: string | null | undefined): string {
  if (!value) return ''
  return value.replace(/[\t\r\n]+/g, ' ').trim()
}

/** ชื่อหมวดในระบบใช้ _ คั่นคำ (เครื่องเขียน_สำนักงาน) แต่ POS ต้องการช่องว่าง */
function formatGroupName(value: string | null | undefined): string {
  return sanitizeField(value).replace(/_/g, ' ')
}

/** POS รับจำนวนเต็มล้วน ไม่มีทศนิยม ไม่มีตัวคั่นหลักพัน */
function formatPrice(price: number | null | undefined): string {
  if (price === null || price === undefined || !Number.isFinite(price)) return ''
  return String(Math.round(price))
}

/**
 * ไล่จากโหนดขึ้นไปหาราก แล้วคืนสายชื่อเรียงจากรากลงมา
 * \seen\ กันไว้เผื่อ parent_id วนกลับมาหาตัวเอง — ข้อมูลเสียไม่ควรทำให้เว็บแฮงก์
 */
export function buildCategoryChain(
  categoryId: string | null | undefined,
  nodesById: Map<string, TaxonomyNodeLite>
): string[] {
  const chain: string[] = []
  const seen = new Set<string>()
  let currentId: string | null = categoryId ?? null

  while (currentId && !seen.has(currentId)) {
    seen.add(currentId)
    const node = nodesById.get(currentId)
    if (!node) break
    chain.unshift(node.name_th)
    currentId = node.parent_id
  }

  return chain
}

/**
 * วางหมวดหมู่ตามความลึกจริง ไม่ใช่ตามคอลัมน์ \level\
 * ตอนนี้ taxonomy มีแค่ 2 ชั้น Sub Group2 จึงว่าง
 * วันไหนเพิ่มชั้น 3 เข้ามา คอลัมน์นั้นจะเติมเองโดยไม่ต้องแก้โค้ด
 */
export function toPosExportRow(
  product: ExportableProduct,
  nodesById: Map<string, TaxonomyNodeLite>
): PosExportRow {
  const chain = buildCategoryChain(product.category_id, nodesById)

  return {
    itemGroup: chain[0] ?? '',
    subGroup1: chain[1] ?? '',
    subGroup2: chain[2] ?? '',
    itemId: product.sku ?? '',
    barcode: product.barcode ?? '',
    itemName: product.name_th,
    unitName: product.unit_name ?? DEFAULT_UNIT,
    price: product.price,
    vat: product.vat_flag ?? DEFAULT_VAT,
  }
}

/**
 * นับช่องที่ POS ต้องการแต่ระบบยังไม่มีข้อมูล
 * ให้ UI เตือนก่อนดาวน์โหลด ดีกว่าปล่อยไฟล์เสียไปให้ POS ปฏิเสธ
 */
export function countPosExportIssues(rows: PosExportRow[]): PosExportIssues {
  const count = (predicate: (row: PosExportRow) => boolean) => rows.filter(predicate).length

  return {
    missingItemId: count((row) => !sanitizeField(row.itemId)),
    // บาร์โค้ดว่างไม่นับว่าขาด ตราบใดที่มี Item ID ให้ fallback ตามไฟล์ต้นแบบ
    missingBarcode: count((row) => !sanitizeField(row.barcode) && !sanitizeField(row.itemId)),
    missingItemName: count((row) => !sanitizeField(row.itemName)),
    missingUnitName: count((row) => !sanitizeField(row.unitName)),
    missingItemGroup: count((row) => !sanitizeField(row.itemGroup)),
    missingPrice: count((row) => formatPrice(row.price) === ''),
  }
}

export function buildPosExportFile(rows: PosExportRow[]): string {
  const lines: string[] = [POS_EXPORT_HEADERS.join(FIELD_SEPARATOR)]

  for (const row of rows) {
    const itemId = sanitizeField(row.itemId)
    lines.push(
      [
        formatGroupName(row.itemGroup),
        formatGroupName(row.subGroup1),
        itemId,
        // ไฟล์ต้นแบบ: แถวที่ไม่มี EAN จริงใช้ Item ID ซ้ำแทน ไม่เว้นว่าง
        sanitizeField(row.barcode) || itemId,
        sanitizeField(row.itemName),
        sanitizeField(row.unitName),
        formatPrice(row.price),
        sanitizeField(row.vat) || DEFAULT_VAT,
        formatGroupName(row.subGroup2),
      ].join(FIELD_SEPARATOR)
    )
  }

  return lines.join(ROW_SEPARATOR) + ROW_SEPARATOR
}
