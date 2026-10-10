# POS Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ให้คนในออฟฟิศกดปุ่มเดียวในหน้าเว็บ แล้วได้ไฟล์ `.txt` 9 คอลัมน์ที่ POS นำเข้าได้ทันที บรรจุเฉพาะสินค้าใหม่ที่ยังไม่มีในสตอก

**Architecture:** แยกเป็นสามชั้น — (1) ฟังก์ชันบริสุทธิ์ใน `utils/pos-export.ts` (ประกอบไฟล์) และ `utils/pos-item-id.ts` (ออกรหัสสินค้า) ทดสอบได้โดยไม่แตะฐานข้อมูล (2) route `/api/pos-export` ที่ดึงข้อมูลด้วย service role เรียกสองโมดูลข้างต้น และเขียนรหัสใหม่กลับลง `products.sku` (3) แท็บ UI ที่เรียก route นั้น การ map หมวดหมู่ทำใน TypeScript ไม่ใช่ SQL ตามกฎ AGENTS.md และเพราะ taxonomy มีแค่ 134 โหนด

การออกรหัสรันทั้งใน `GET` (ดูตัวอย่าง ไม่เขียนอะไร) และ `POST` (เขียนจริง) ด้วยตรรกะเดียวกันและเรียงลำดับเดิมเสมอ รหัสที่เห็นบนจอจึงตรงกับไฟล์ที่ดาวน์โหลดได้

**Tech Stack:** Next.js App Router, TypeScript, Supabase (service role client), Zod, Jest, Playwright, PostgreSQL migration

**Spec:** [docs/specs/2026-09-06-pos-export.md](../../specs/2026-09-06-pos-export.md) — อ่านก่อนเริ่ม โดยเฉพาะหัวข้อหัวตารางที่สะกดผิด

## Global Constraints

- **หัวคอลัมน์ที่ 5 สะกดผิดในต้นฉบับ ห้ามแก้** — `Item Name(` + `\u0E0A\u0E37\u0E48\u0E48\u0E2D\u0E2A\u0E34\u0E19\u0E04\u0E49\u0E32` + `)` มีไม้เอก U+0E48 **สองตัวติดกัน** เขียนในโค้ดด้วย escape เสมอ ห้ามพิมพ์ตัวอักษรตรง ๆ
- ไฟล์ที่ออก: tab-separated, UTF-8 **ไม่มี BOM**, ขึ้นบรรทัด **CRLF** รวมบรรทัดสุดท้าย, **ไม่ครอบด้วยเครื่องหมายคำพูดใด ๆ**
- คัดแถวด้วย `metadata->>'source' = 'import_wizard'` **และ** `metadata->>'dedup_bucket' = 'new'` **และ** `pos_exported_at IS NULL` — **ห้ามกรองด้วย `status`** เพราะ verify พลิกเป็น `approved` ปนกับสตอกเดิม
- ราคาเขียนเป็นจำนวนเต็มล้วน ไม่มีทศนิยม ไม่มีตัวคั่นหลักพัน
- ชื่อหมวดหมู่แทน `_` ด้วยช่องว่างตอนเขียนไฟล์ **ห้ามแก้ในฐานข้อมูล** (ผูกกับ `keyword_rules` ที่ mine ไว้)
- **POS บังคับ 6 ช่อง**: `Item ID`, `Barcode`, `Item Name`, `Price`, `Unit Name`, `Item Group` — ตัวส่งออกต้องนับแถวที่ขาดครบทั้ง 6 ช่อง ไม่ใช่แค่บางช่อง
- `Item ID` **ออกอัตโนมัติ** จาก prefix ตามหมวดหลัก (`HH`, `PC`, `KIT`, …) เดินเลขต่อจากเลขสูงสุดที่ใช้ไปแล้ว — และ **เขียนลง `products.sku` ตอนดาวน์โหลด** เพื่อให้รหัสนิ่งตลอดไป
- `Unit Name` ที่ว่างเติม `ชิ้น` **ตอนประกอบไฟล์** ไม่ใช่ default ในฐานข้อมูล
- ไฟล์คงไว้ **9 คอลัมน์ตามต้นแบบ** แม้ POS จะ map `Sub Group1`/`Sub Group2` ไม่ได้
- `products.sku` มี `UNIQUE` constraint อยู่แล้ว — คำนวณเลขสูงสุดใหม่ทุกครั้งที่แจกรหัส ห้ามใช้ค่าที่แคชไว้
- TypeScript ห้ามใช้ `any` · ชื่อตัวแปร `camelCase` · ห้าม raw SQL ในโค้ดแอป
- Tailwind ใช้ได้เฉพาะใน `taxonomy-app/` เท่านั้น
- ทุกคำสั่งรันจาก `taxonomy-app/` เว้นแต่ระบุเป็นอย่างอื่น

---

### Task 1: ฟังก์ชันสร้างไฟล์ (ล้วน ๆ ไม่แตะฐานข้อมูล)

หัวใจของงานทั้งหมด แยกออกมาเพราะทดสอบได้เร็วและไม่ต้องมีฐานข้อมูล ทำให้ตรงนี้ถูกก่อน แล้วที่เหลือเป็นแค่ท่อส่งข้อมูล

**Files:**
- Create: `taxonomy-app/utils/pos-export.ts`
- Test: `taxonomy-app/__tests__/utils/pos-export.test.ts`

**Interfaces:**
- Consumes: ไม่มี — งานแรก
- Produces:
  - `POS_EXPORT_HEADERS: readonly string[]` (9 ตัว)
  - `interface TaxonomyNodeLite { id: string; name_th: string; parent_id: string | null }`
  - `interface ExportableProduct { name_th: string; sku: string | null; barcode: string | null; unit_name: string | null; vat_flag: string | null; price: number | null; category_id: string | null }`
  - `interface PosExportRow { itemGroup: string; subGroup1: string; subGroup2: string; itemId: string; barcode: string; itemName: string; unitName: string; price: number | null; vat: string }`
  - `interface PosExportIssues { missingItemId: number; missingBarcode: number; missingItemName: number; missingUnitName: number; missingItemGroup: number; missingPrice: number }` — หนึ่งช่องต่อหนึ่งช่องบังคับของ POS
  - `DEFAULT_UNIT: string` (`'ชิ้น'`)
  - `buildCategoryChain(categoryId: string | null | undefined, nodesById: Map<string, TaxonomyNodeLite>): string[]`
  - `toPosExportRow(product: ExportableProduct, nodesById: Map<string, TaxonomyNodeLite>): PosExportRow`
  - `countPosExportIssues(rows: PosExportRow[]): PosExportIssues`
  - `buildPosExportFile(rows: PosExportRow[]): string`

- [ ] **Step 1: เขียนเทสต์ที่ยังไม่ผ่าน**

สร้าง `taxonomy-app/__tests__/utils/pos-export.test.ts`:

```ts
/**
 * ไฟล์ที่ POS กินได้มีเงื่อนไขที่มองไม่เห็นด้วยตาหลายข้อ — ไม้เอกซ้ำในหัวคอลัมน์,
 * CRLF, ไม่มี BOM, ไม่มีเครื่องหมายคำพูด เทสต์ชุดนี้ล็อกทุกข้อไว้
 * สเปกเต็ม: docs/specs/2026-09-06-pos-export.md
 */
import {
  POS_EXPORT_HEADERS,
  buildCategoryChain,
  buildPosExportFile,
  countPosExportIssues,
  toPosExportRow,
  type ExportableProduct,
  type PosExportRow,
  type TaxonomyNodeLite,
} from '@/utils/pos-export'

const nodes = (...list: TaxonomyNodeLite[]) =>
  new Map<string, TaxonomyNodeLite>(list.map((n) => [n.id, n]))

const product = (overrides: Partial<ExportableProduct> = {}): ExportableProduct => ({
  name_th: 'ตะกร้ากลม 9899',
  sku: 'HH001',
  barcode: null,
  unit_name: 'ใบ',
  vat_flag: 'V',
  price: 45,
  category_id: 'sub',
  ...overrides,
})

const row = (overrides: Partial<PosExportRow> = {}): PosExportRow => ({
  itemGroup: 'ของใช้ในบ้าน',
  subGroup1: 'ตะกร้า/กระจาด',
  subGroup2: '',
  itemId: 'HH001',
  barcode: 'HH001',
  itemName: 'ตะกร้ากลม 9899',
  unitName: 'ใบ',
  price: 45,
  vat: 'V',
  ...overrides,
})

describe('หัวตาราง', () => {
  it('มี 9 คอลัมน์', () => {
    expect(POS_EXPORT_HEADERS).toHaveLength(9)
  })

  it('คอลัมน์ที่ 5 คงไม้เอกซ้ำสองตัวตามต้นฉบับ (ห้ามแก้คำผิด)', () => {
    // U+0E48 สองตัวติดกัน — ถ้ามีใคร "แก้ให้" POS จะจับคอลัมน์ไม่เจอ
    expect(POS_EXPORT_HEADERS[4]).toBe(
      'Item Name(\u0E0A\u0E37\u0E48\u0E48\u0E2D\u0E2A\u0E34\u0E19\u0E04\u0E49\u0E32)'
    )
    const codepoints = Array.from(POS_EXPORT_HEADERS[4]).map((c) => c.codePointAt(0))
    expect(codepoints.filter((c) => c === 0x0e48)).toHaveLength(2)
  })

  it('อีก 8 คอลัมน์ตรงตามต้นฉบับ', () => {
    expect(POS_EXPORT_HEADERS[0]).toBe('Item Group(ชื่อกลุ่มสินค้า)')
    expect(POS_EXPORT_HEADERS[1]).toBe('Sub Group1(ชื่อกลุ่มย่อย1)')
    expect(POS_EXPORT_HEADERS[2]).toBe('Item ID(รหัสสินค้า)')
    expect(POS_EXPORT_HEADERS[3]).toBe('Barcode(บาร์โค้ด)')
    expect(POS_EXPORT_HEADERS[5]).toBe('Unit Name(ชื่อหน่วย)')
    expect(POS_EXPORT_HEADERS[6]).toBe('Price(ราคาขาย)')
    expect(POS_EXPORT_HEADERS[7]).toBe('Vat')
    expect(POS_EXPORT_HEADERS[8]).toBe('Sub Group2')
  })
})

describe('buildCategoryChain', () => {
  const tree = nodes(
    { id: 'root', name_th: 'ของใช้ในบ้าน', parent_id: null },
    { id: 'sub', name_th: 'ตะกร้า/กระจาด', parent_id: 'root' },
    { id: 'leaf', name_th: 'ตะกร้าผ้า', parent_id: 'sub' }
  )

  it('คืนสายเรียงจากรากลงมา', () => {
    expect(buildCategoryChain('leaf', tree)).toEqual([
      'ของใช้ในบ้าน', 'ตะกร้า/กระจาด', 'ตะกร้าผ้า',
    ])
  })

  it('คืนอาเรย์ว่างเมื่อไม่มี category_id', () => {
    expect(buildCategoryChain(null, tree)).toEqual([])
  })

  it('คืนอาเรย์ว่างเมื่อ id ไม่มีในต้นไม้', () => {
    expect(buildCategoryChain('ไม่มีจริง', tree)).toEqual([])
  })

  it('ไม่แฮงก์เมื่อ parent วนกลับมาหาตัวเอง', () => {
    const cyclic = nodes(
      { id: 'a', name_th: 'A', parent_id: 'b' },
      { id: 'b', name_th: 'B', parent_id: 'a' }
    )
    expect(buildCategoryChain('a', cyclic)).toEqual(['B', 'A'])
  })
})

describe('toPosExportRow', () => {
  const tree = nodes(
    { id: 'root', name_th: 'ของใช้ในบ้าน', parent_id: null },
    { id: 'sub', name_th: 'ตะกร้า/กระจาด', parent_id: 'root' }
  )

  it('วางหมวดตามความลึก — ราก=Item Group, ชั้น2=Sub Group1', () => {
    const result = toPosExportRow(product(), tree)
    expect(result.itemGroup).toBe('ของใช้ในบ้าน')
    expect(result.subGroup1).toBe('ตะกร้า/กระจาด')
    expect(result.subGroup2).toBe('')
  })

  it('เติม Sub Group2 เมื่อ taxonomy มีชั้น 3', () => {
    const deep = nodes(
      { id: 'root', name_th: 'ของใช้ในบ้าน', parent_id: null },
      { id: 'sub', name_th: 'ตะกร้า/กระจาด', parent_id: 'root' },
      { id: 'leaf', name_th: 'ตะกร้าผ้า', parent_id: 'sub' }
    )
    expect(toPosExportRow(product({ category_id: 'leaf' }), deep).subGroup2).toBe('ตะกร้าผ้า')
  })

  it('หมวดที่เป็นรากอยู่แล้วได้แค่ Item Group ไม่ crash', () => {
    const result = toPosExportRow(product({ category_id: 'root' }), tree)
    expect(result.itemGroup).toBe('ของใช้ในบ้าน')
    expect(result.subGroup1).toBe('')
  })

  it('vat_flag ที่เป็น null ตกลงมาเป็น V', () => {
    expect(toPosExportRow(product({ vat_flag: null }), tree).vat).toBe('V')
  })

  it('unit_name ที่เป็น null ตกลงมาเป็น ชิ้น (POS บังคับช่องนี้)', () => {
    expect(toPosExportRow(product({ unit_name: null }), tree).unitName).toBe('ชิ้น')
  })

  it('unit_name ที่กรอกไว้จริงชนะค่าเริ่มต้น', () => {
    expect(toPosExportRow(product({ unit_name: 'คู่' }), tree).unitName).toBe('คู่')
  })
})

describe('buildPosExportFile', () => {
  it('ขึ้นต้นด้วยหัวตารางคั่นด้วย tab', () => {
    const text = buildPosExportFile([])
    expect(text.split('\r\n')[0]).toBe(POS_EXPORT_HEADERS.join('\t'))
  })

  it('ใช้ CRLF และปิดท้ายไฟล์ด้วย CRLF', () => {
    const text = buildPosExportFile([row()])
    expect(text.endsWith('\r\n')).toBe(true)
    expect(text.split('\r\n').filter(Boolean)).toHaveLength(2)
    expect(/[^\r]\n/.test(text)).toBe(false)
  })

  it('ไม่มี BOM นำหน้า', () => {
    expect(buildPosExportFile([row()]).charCodeAt(0)).not.toBe(0xfeff)
  })

  it('ไม่ครอบค่าด้วยเครื่องหมายคำพูด แม้ชื่อจะมีจุลภาค', () => {
    const text = buildPosExportFile([row({ itemName: 'ตะกร้า, กลม' })])
    expect(text).not.toContain('"')
    expect(text).toContain('ตะกร้า, กลม')
  })

  it('ทุกแถวมี 9 ช่องเท่ากับหัวตาราง', () => {
    const text = buildPosExportFile([row(), row({ itemId: 'HH002' })])
    for (const line of text.split('\r\n').filter(Boolean)) {
      expect(line.split('\t')).toHaveLength(9)
    }
  })

  it('แทน _ ในชื่อหมวดด้วยช่องว่าง', () => {
    const text = buildPosExportFile([row({ itemGroup: 'เครื่องเขียน_สำนักงาน' })])
    expect(text).toContain('เครื่องเขียน สำนักงาน')
    expect(text).not.toContain('เครื่องเขียน_สำนักงาน')
  })

  it('เขียนราคาเป็นจำนวนเต็มล้วน', () => {
    const fields = buildPosExportFile([row({ price: 339.0 })]).split('\r\n')[1].split('\t')
    expect(fields[6]).toBe('339')
  })

  it('ปัดเศษราคาที่มีทศนิยม', () => {
    const fields = buildPosExportFile([row({ price: 45.6 })]).split('\r\n')[1].split('\t')
    expect(fields[6]).toBe('46')
  })

  it('ราคาที่ยังไม่มีเขียนเป็นช่องว่าง ไม่ใช่ null', () => {
    const fields = buildPosExportFile([row({ price: null })]).split('\r\n')[1].split('\t')
    expect(fields[6]).toBe('')
  })

  it('ไม่มีบาร์โค้ดให้ใช้ Item ID แทน ตามไฟล์ต้นแบบ', () => {
    const fields = buildPosExportFile([row({ barcode: '' })]).split('\r\n')[1].split('\t')
    expect(fields[3]).toBe('HH001')
  })

  it('มีบาร์โค้ดจริงก็ใช้ตัวจริง', () => {
    const fields = buildPosExportFile([row({ barcode: '8859358411653' })]).split('\r\n')[1].split('\t')
    expect(fields[3]).toBe('8859358411653')
  })

  it('tab หรือ newline ที่หลุดมาในชื่อสินค้าไม่ทำให้โครงไฟล์พัง', () => {
    const text = buildPosExportFile([row({ itemName: 'ตะกร้า\tกลม\r\nใหญ่' })])
    expect(text.split('\r\n').filter(Boolean)).toHaveLength(2)
    expect(text.split('\r\n')[1].split('\t')).toHaveLength(9)
  })
})

describe('countPosExportIssues', () => {
  it('ไฟล์ที่ครบถ้วนได้ศูนย์ทุกช่อง', () => {
    expect(countPosExportIssues([row()])).toEqual({
      missingItemId: 0, missingBarcode: 0, missingItemName: 0,
      missingUnitName: 0, missingItemGroup: 0, missingPrice: 0,
    })
  })

  it('นับครบทั้ง 6 ช่องบังคับของ POS', () => {
    const issues = countPosExportIssues([
      row(),
      row({ itemId: '', barcode: '', price: null }),
      row({ itemName: '', unitName: '', itemGroup: '', price: null }),
    ])
    expect(issues).toEqual({
      missingItemId: 1, missingBarcode: 1, missingItemName: 1,
      missingUnitName: 1, missingItemGroup: 1, missingPrice: 2,
    })
  })

  it('บาร์โค้ดว่างแต่มี Item ID ไม่นับว่าขาด เพราะไฟล์ fallback ให้', () => {
    expect(countPosExportIssues([row({ barcode: '' })]).missingBarcode).toBe(0)
  })
})
```

- [ ] **Step 2: รันเทสต์ให้เห็นว่าไม่ผ่าน**

Run: `npx jest __tests__/utils/pos-export.test.ts`
Expected: FAIL — `Cannot find module '@/utils/pos-export'`

- [ ] **Step 3: เขียนโค้ดให้ผ่าน**

สร้าง `taxonomy-app/utils/pos-export.ts`:

```ts
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
 * `seen` กันไว้เผื่อ parent_id วนกลับมาหาตัวเอง — ข้อมูลเสียไม่ควรทำให้เว็บแฮงก์
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
 * วางหมวดหมู่ตามความลึกจริง ไม่ใช่ตามคอลัมน์ `level`
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
```

- [ ] **Step 4: รันเทสต์ให้ผ่าน**

Run: `npx jest __tests__/utils/pos-export.test.ts`
Expected: PASS ทุกเคส

- [ ] **Step 5: เทียบกับไฟล์ต้นแบบจริง**

Run จากรากรีโป:

```bash
node -e '
const { buildPosExportFile, POS_EXPORT_HEADERS } = require("./taxonomy-app/utils/pos-export.ts");
' 2>/dev/null || npx tsx -e '
import { POS_EXPORT_HEADERS } from "./taxonomy-app/utils/pos-export"
import { readFileSync } from "fs"
const sample = "D:/OneDrive/เอกสาร/pos/POSไฟล์นำเข้า/นำเข้าสินค้า/รองเท้าขัดมันหญิงแชปปี้.txt"
const header = readFileSync(sample, "utf8").split("\r\n")[0]
const ours = POS_EXPORT_HEADERS.join("\t")
console.log(header === ours ? "MATCH" : "MISMATCH")
if (header !== ours) { console.log("sample:", JSON.stringify(header)); console.log("ours  :", JSON.stringify(ours)) }
'
```

Expected: `MATCH`
ถ้า `tsx` ไม่มีในเครื่อง ข้ามขั้นนี้ได้ — เทสต์ใน Step 1 ล็อก codepoint ไว้แล้ว

- [ ] **Step 6: Commit**

```bash
git add taxonomy-app/utils/pos-export.ts taxonomy-app/__tests__/utils/pos-export.test.ts docs/specs/2026-09-06-pos-export.md
git commit -m "feat: add POS export file builder

หัวคอลัมน์ที่ 5 เขียนด้วย escape เพราะต้นฉบับสะกดผิด (ไม้เอกซ้ำสองตัว)
และห้ามแก้ ไม่งั้น POS จับคอลัมน์ไม่เจอ

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Migration เพิ่มคอลัมน์ที่ POS ต้องการ

**Files:**
- Create: `taxonomy-app/supabase/migrations/20260906000000_add_pos_export_columns.sql`

**Interfaces:**
- Consumes: ไม่มี
- Produces: `products.barcode`, `products.unit_name`, `products.vat_flag`, `products.pos_exported_at` — Task 3 อ่านทั้งสี่คอลัมน์

- [ ] **Step 1: เขียน migration**

สร้าง `taxonomy-app/supabase/migrations/20260906000000_add_pos_export_columns.sql`:

```sql
-- คอลัมน์ที่ไฟล์นำเข้า POS ต้องการ แต่ products ยังไม่มี (2026-09-06)
--
-- ไฟล์ที่ POS กินได้มี 9 คอลัมน์ ตอนนี้ตารางตอบได้แค่ 5
-- (ชื่อ, รหัส, ราคา, และหมวด 2 ชั้นผ่าน category_id) ที่เหลือต้องเพิ่มเข้ามา
--
-- pos_exported_at ไม่ได้มีไว้ทำรายงาน แต่เป็นตัวกันส่งออกซ้ำ:
-- ของที่ส่งเข้า POS แล้วก็คือ "มีในสตอกแล้ว" ไม่ควรโผล่ในไฟล์รอบหน้าอีก
--
-- สเปก: docs/specs/2026-09-06-pos-export.md

ALTER TABLE products
    ADD COLUMN IF NOT EXISTS barcode         TEXT,
    ADD COLUMN IF NOT EXISTS unit_name       TEXT,
    ADD COLUMN IF NOT EXISTS vat_flag        TEXT NOT NULL DEFAULT 'V',
    ADD COLUMN IF NOT EXISTS pos_exported_at TIMESTAMPTZ;

COMMENT ON COLUMN products.barcode IS
    'บาร์โค้ดสำหรับ POS — คนละอย่างกับ sku ถ้าว่าง ตัวส่งออกจะใช้ sku แทนตามไฟล์ต้นแบบ';
COMMENT ON COLUMN products.unit_name IS
    'หน่วยนับที่ POS แสดง เช่น ชิ้น / คู่ / กล่อง';
COMMENT ON COLUMN products.vat_flag IS
    'ธง VAT ในไฟล์นำเข้า POS — ไฟล์ต้นแบบทุกแถวเป็น V จึงตั้ง default ไว้';
COMMENT ON COLUMN products.pos_exported_at IS
    'เวลาที่ถูกใส่ในไฟล์นำเข้า POS ครั้งล่าสุด — NULL แปลว่ายังไม่เคยส่งออก ใช้กันส่งซ้ำ';

-- ตัวส่งออกคัดด้วยสามเงื่อนไขนี้เสมอ (source + dedup_bucket + ยังไม่ส่งออก)
-- partial index เพราะแถวที่ส่งออกไปแล้วไม่ต้องอยู่ในดัชนี
CREATE INDEX IF NOT EXISTS idx_products_pending_pos_export
    ON products ((metadata->>'source'), (metadata->>'dedup_bucket'))
    WHERE pos_exported_at IS NULL;
```

- [ ] **Step 2: รัน migration**

Run จากรากรีโป:

```bash
npx supabase db push --db-url "postgresql://postgres:postgres@127.0.0.1:54325/postgres"
```

ถ้าคำสั่งนี้ใช้ไม่ได้ในเครื่องนี้ ให้รันไฟล์ตรง ๆ:

```bash
psql "postgresql://postgres:postgres@127.0.0.1:54325/postgres" -f taxonomy-app/supabase/migrations/20260906000000_add_pos_export_columns.sql
```

- [ ] **Step 3: ยืนยันว่าคอลัมน์เข้าจริง**

Run:

```bash
psql "postgresql://postgres:postgres@127.0.0.1:54325/postgres" -c "SELECT column_name, data_type, column_default FROM information_schema.columns WHERE table_name='products' AND column_name IN ('barcode','unit_name','vat_flag','pos_exported_at') ORDER BY column_name;"
```

Expected: 4 แถว และ `vat_flag` มี default `'V'::text`

- [ ] **Step 4: อัปเดต type ฝั่ง TypeScript**

แก้ `taxonomy-app/utils/supabase.ts` — เพิ่มสี่ฟิลด์เข้า `interface Product` (อยู่บรรทัด 100-123) ต่อจาก `price`:

```ts
  price?: number
  barcode?: string
  unit_name?: string
  vat_flag?: string
  pos_exported_at?: string
```

- [ ] **Step 5: typecheck**

Run: `npx tsc --noEmit`
Expected: ยังคง **2 errors เท่าเดิม** ทั้งคู่อยู่ใน `__tests__/integration/synonym.integration.test.ts` (ค้างมาก่อนงานนี้) — ถ้าเลข error เพิ่ม แปลว่างานนี้ทำพัง ต้องแก้ก่อนไปต่อ

- [ ] **Step 6: Commit**

```bash
git add taxonomy-app/supabase/migrations/20260906000000_add_pos_export_columns.sql taxonomy-app/utils/supabase.ts
git commit -m "feat: add barcode, unit_name, vat_flag, pos_exported_at to products

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: ตัวออกรหัสสินค้า

`Item ID` เป็นช่องบังคับของ POS แต่สินค้าใหม่ทั้ง 221 รายการไม่มี `sku` เลย และไฟล์ต้นทางก็ไม่มีคอลัมน์รหัส

สตอกเดิม 3,103 รายการใช้รูปแบบ `<PREFIX><เลขรัน>` โดย prefix ผูกกับหมวดหลักแบบ 1:1 (100% ใน 15 จาก 16 หมวด) และเลขเดินต่อเนื่องไม่มีช่องว่าง งานนี้ออกรหัสใหม่ให้เข้าชุดเดียวกัน

**Files:**
- Create: `taxonomy-app/utils/pos-item-id.ts`
- Test: `taxonomy-app/__tests__/utils/pos-item-id.test.ts`

**Interfaces:**
- Consumes: ไม่มี — ฟังก์ชันบริสุทธิ์ทั้งหมด ไม่แตะฐานข้อมูล
- Produces:
  - `MAIN_CATEGORY_PREFIX: Record<string, string>` (16 หมวด)
  - `FALLBACK_PREFIX: string` (`'MIS'`)
  - `interface ItemIdCandidate { id: string; name_th: string; sku: string | null; mainCategory: string | null }`
  - `prefixForMainCategory(mainCategory: string | null | undefined): string`
  - `maxNumberByPrefix(existingSkus: readonly string[]): Map<string, number>`
  - `allocateItemIds(candidates: readonly ItemIdCandidate[], existingSkus: readonly string[]): Map<string, string>` — คีย์เป็น `product.id` ค่าเป็นรหัสใหม่ มีเฉพาะรายการที่ยังไม่มี `sku`

- [ ] **Step 1: เขียนเทสต์ที่ยังไม่ผ่าน**

สร้าง `taxonomy-app/__tests__/utils/pos-item-id.test.ts`:

```ts
/**
 * รหัสที่แจกไปแล้วจะไปอยู่ใน POS ถาวร แก้ทีหลังไม่ได้ง่าย ๆ
 * เทสต์ชุดนี้จึงล็อกสามอย่าง: เข้าชุดกับของเดิม, ไม่ชนกันเอง, และรันซ้ำได้ผลเดิม
 */
import {
  FALLBACK_PREFIX,
  MAIN_CATEGORY_PREFIX,
  allocateItemIds,
  maxNumberByPrefix,
  prefixForMainCategory,
  type ItemIdCandidate,
} from '@/utils/pos-item-id'

const candidate = (over: Partial<ItemIdCandidate> & { id: string }): ItemIdCandidate => ({
  name_th: 'สินค้า ' + over.id,
  sku: null,
  mainCategory: 'ของใช้ในบ้าน',
  ...over,
})

describe('prefixForMainCategory', () => {
  it('map หมวดหลักที่รู้จักตามที่สตอกเดิมใช้', () => {
    expect(prefixForMainCategory('ของใช้ในบ้าน')).toBe('HH')
    expect(prefixForMainCategory('ผลิตภัณฑ์ดูแลส่วนบุคคล')).toBe('PC')
    expect(prefixForMainCategory('เครื่องครัว')).toBe('KIT')
    expect(prefixForMainCategory('เครื่องมือ_ฮาร์ดแวร์')).toBe('HW')
    expect(prefixForMainCategory('เครื่องเขียน_สำนักงาน')).toBe('ST')
  })

  it('ครบทั้ง 16 หมวด และ prefix ไม่ซ้ำกัน', () => {
    const prefixes = Object.values(MAIN_CATEGORY_PREFIX)
    expect(prefixes).toHaveLength(16)
    expect(new Set(prefixes).size).toBe(16)
  })

  it('หมวดที่ไม่รู้จักหรือไม่มีหมวด ตกไปใช้ MIS', () => {
    expect(prefixForMainCategory('หมวดที่ยังไม่มี')).toBe(FALLBACK_PREFIX)
    expect(prefixForMainCategory(null)).toBe(FALLBACK_PREFIX)
    expect(prefixForMainCategory('')).toBe(FALLBACK_PREFIX)
  })
})

describe('maxNumberByPrefix', () => {
  it('อ่านเลขสูงสุดของแต่ละ prefix', () => {
    const maxes = maxNumberByPrefix(['HH001', 'HH695', 'PC754', 'HH042'])
    expect(maxes.get('HH')).toBe(695)
    expect(maxes.get('PC')).toBe(754)
  })

  it('ข้ามรหัสที่ไม่เข้ารูปแบบ ไม่พังทั้งชุด', () => {
    const maxes = maxNumberByPrefix(['HH010', 'SS-CHP-BK-34', '', 'ไทย123'])
    expect(maxes.get('HH')).toBe(10)
    expect(maxes.size).toBe(1)
  })
})

describe('allocateItemIds', () => {
  it('เดินเลขต่อจากตัวสูงสุดที่ใช้ไปแล้ว', () => {
    const assigned = allocateItemIds([candidate({ id: 'a' })], ['HH695', 'PC754'])
    expect(assigned.get('a')).toBe('HH696')
  })

  it('แจกเรียงไม่ซ้ำกันเองภายในหมวดเดียวกัน', () => {
    const assigned = allocateItemIds(
      [candidate({ id: 'a', name_th: 'ก' }), candidate({ id: 'b', name_th: 'ข' }), candidate({ id: 'c', name_th: 'ค' })],
      ['HH695']
    )
    expect([...assigned.values()].sort()).toEqual(['HH696', 'HH697', 'HH698'])
    expect(new Set(assigned.values()).size).toBe(3)
  })

  it('แต่ละหมวดเดินเลขของตัวเอง', () => {
    const assigned = allocateItemIds(
      [
        candidate({ id: 'a', mainCategory: 'ของใช้ในบ้าน' }),
        candidate({ id: 'b', mainCategory: 'เครื่องครัว' }),
      ],
      ['HH695', 'KIT413']
    )
    expect(assigned.get('a')).toBe('HH696')
    expect(assigned.get('b')).toBe('KIT414')
  })

  it('ข้ามรายการที่มีรหัสอยู่แล้ว ไม่ออกรหัสทับ', () => {
    const assigned = allocateItemIds(
      [candidate({ id: 'a', sku: 'HH001' }), candidate({ id: 'b' })],
      ['HH695']
    )
    expect(assigned.has('a')).toBe(false)
    expect(assigned.get('b')).toBe('HH696')
  })

  it('ไม่ชนรหัสที่มีอยู่แล้วแม้เลขจะขาดช่วง', () => {
    // HH696 ถูกใช้ไปแล้วทั้งที่ max คือ 695 ไม่ได้ (ข้อมูลจริงอาจไม่เรียบร้อย)
    const assigned = allocateItemIds([candidate({ id: 'a' })], ['HH695', 'HH696'])
    expect(assigned.get('a')).toBe('HH697')
  })

  it('รันซ้ำด้วยข้อมูลเดิมได้ผลเดิมทุกครั้ง (preview ต้องตรงกับตอนโหลดจริง)', () => {
    const list = [
      candidate({ id: 'b', name_th: 'ข' }),
      candidate({ id: 'a', name_th: 'ก' }),
      candidate({ id: 'c', name_th: 'ค' }),
    ]
    const first = allocateItemIds(list, ['HH695'])
    const second = allocateItemIds([...list].reverse(), ['HH695'])
    expect([...first.entries()].sort()).toEqual([...second.entries()].sort())
  })

  it('ไม่มีรายการที่ต้องแจกก็คืน map ว่าง', () => {
    expect(allocateItemIds([], ['HH695']).size).toBe(0)
  })
})
```

- [ ] **Step 2: รันเทสต์ให้เห็นว่าไม่ผ่าน**

Run: `npx jest __tests__/utils/pos-item-id.test.ts`
Expected: FAIL — `Cannot find module '@/utils/pos-item-id'`

- [ ] **Step 3: เขียนโค้ดให้ผ่าน**

สร้าง `taxonomy-app/utils/pos-item-id.ts`:

```ts
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
```

- [ ] **Step 4: รันเทสต์ให้ผ่าน**

Run: `npx jest __tests__/utils/pos-item-id.test.ts`
Expected: PASS ทุกเคส

- [ ] **Step 5: ตรวจตาราง prefix กับข้อมูลจริง**

Run จากรากรีโป:

```bash
node -e "
const { Client } = require('./taxonomy-app/node_modules/pg');
const c = new Client('postgresql://postgres:postgres@127.0.0.1:54325/postgres');
c.connect().then(() => c.query(\`
  SELECT par.name_th AS cat, substring(p.sku from '^[A-Za-z]+') AS prefix, count(*) n
  FROM products p
  JOIN taxonomy_nodes ch ON ch.id = p.category_id
  JOIN taxonomy_nodes par ON par.id = ch.parent_id
  WHERE p.sku ~ '^[A-Za-z]+[0-9]+$'
  GROUP BY 1,2 ORDER BY 1, 3 DESC
\`)).then(r => { r.rows.forEach(x => console.log(x.cat, '->', x.prefix, x.n)); return c.end(); });
"
```

Expected: หมวดหลักแต่ละหมวดมี prefix เด่นตัวเดียว และตรงกับตารางใน `MAIN_CATEGORY_PREFIX`
ถ้ามีหมวดไหนไม่ตรง ให้แก้ตารางในโค้ดตามข้อมูลจริง อย่าแก้ข้อมูล

- [ ] **Step 6: Commit**

```bash
git add taxonomy-app/utils/pos-item-id.ts taxonomy-app/__tests__/utils/pos-item-id.test.ts
git commit -m "feat: add POS item ID allocator

ออกรหัสตาม prefix ของหมวดหลัก เดินเลขต่อจากสตอกเดิม 3,103 รายการ
เรียงตามชื่อก่อนแจกเพื่อให้หน้าตัวอย่างกับไฟล์จริงได้รหัสตรงกัน

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: API route — ดูตัวอย่างก่อนโหลด (GET)

**Files:**
- Create: `taxonomy-app/app/api/pos-export/route.ts`
- Modify: `taxonomy-app/middleware.ts:18` — เพิ่ม `/api/pos-export` เข้า `GATED_READ_PATHS`
- Test: `taxonomy-app/__tests__/api/pos-export.test.ts`

**Interfaces:**
- Consumes: `buildCategoryChain`, `countPosExportIssues`, `toPosExportRow`, `TaxonomyNodeLite`, `ExportableProduct` จาก Task 1 · คอลัมน์จาก Task 2 · `allocateItemIds` จาก Task 3
- Produces:
  - `GET /api/pos-export` → `{ success: true, total: number, issues: PosExportIssues, preview: PosExportRow[] }`
  - ฟังก์ชันภายใน `loadExportRows(): Promise<{ ids: string[]; rows: PosExportRow[]; assignedSkus: Map<string, string> }>` — Task 5 เรียกซ้ำและใช้ `assignedSkus` เขียนลงฐานข้อมูล

> **สำคัญ:** route นี้อ่านด้วย service role ซึ่งข้าม RLS ทั้งหมด `middleware.ts` ปล่อย GET ผ่านโดยไม่ตรวจ cookie เว้นแต่ path อยู่ใน `GATED_READ_PATHS` — ถ้าลืมเพิ่ม ใครก็ตามที่เปิดถึงเครื่องนี้ได้จะอ่านรายการสินค้าทั้งหมดโดยไม่ต้องปลดล็อก

- [ ] **Step 1: เขียนเทสต์ที่ยังไม่ผ่าน**

สร้าง `taxonomy-app/__tests__/api/pos-export.test.ts`:

```ts
/**
 * route นี้อ่านด้วย service role (ข้าม RLS) จึงต้องอยู่ใน GATED_READ_PATHS
 * ของ middleware ไม่งั้น GET จะเปิดให้อ่านได้โดยไม่ต้องปลดล็อก
 *
 * @jest-environment node
 */
import { NextRequest } from 'next/server'
import { readFileSync } from 'fs'
import path from 'path'

type Filter = { kind: string; column?: string; value?: unknown }
type Call = { table: string; op: string; payload?: unknown; filters: Filter[] }

const calls: Call[] = []

/**
 * คิวผลลัพธ์ต่อ `table.op` เพราะ route ยิง products สองครั้ง:
 * ครั้งแรกเอารายการที่จะส่งออก ครั้งที่สองเอา sku ทั้งหมดไปคำนวณรหัสใหม่
 */
let queues: Record<string, Array<{ data?: unknown; error?: unknown }>> = {}

const resultFor = (table: string, op: string) =>
  queues[`${table}.${op}`]?.shift() ?? { data: [], error: null }

const makeChain = (table: string, op: string, payload?: unknown) => {
  const call: Call = { table, op, payload, filters: [] }
  calls.push(call)
  const settled = resultFor(table, op)
  const chain: Record<string, unknown> = {}
  const self = () => chain
  Object.assign(chain, {
    select: self,
    order: self,
    not: self,
    eq: (column: string, value: unknown) => {
      call.filters.push({ kind: 'eq', column, value })
      return chain
    },
    is: (column: string, value: unknown) => {
      call.filters.push({ kind: 'is', column, value })
      return chain
    },
    in: (column: string, value: unknown) => {
      call.filters.push({ kind: 'in', column, value })
      return Promise.resolve(settled)
    },
    then: (resolve: (v: unknown) => unknown) => Promise.resolve(settled).then(resolve),
  })
  return chain
}

jest.mock('@/utils/supabase-admin', () => ({
  supabaseAdmin: {
    from: (table: string) => ({
      select: (...args: unknown[]) => makeChain(table, 'select', args),
      update: (payload: unknown) => makeChain(table, 'update', payload),
    }),
  },
}))

const PRODUCTS = [
  {
    id: 'p1',
    name_th: 'ตะกร้ากลม 9899',
    sku: 'HH001',
    barcode: null,
    unit_name: 'ใบ',
    vat_flag: 'V',
    price: 45,
    category_id: 'sub',
  },
  {
    id: 'p2',
    name_th: 'เทปปะหม้อม้วนเล็ก',
    sku: null,
    barcode: null,
    unit_name: null,
    vat_flag: 'V',
    price: null,
    category_id: 'sub',
  },
]

const NODES = [
  { id: 'root', name_th: 'เครื่องเขียน_สำนักงาน', parent_id: null },
  { id: 'sub', name_th: 'กาว/เทป', parent_id: 'root' },
]

/** sku ที่สตอกเดิมใช้ไปแล้ว — ตัวออกรหัสต้องเดินเลขต่อจากนี่ */
const EXISTING_SKUS = [{ sku: 'ST133' }, { sku: 'HH695' }]

beforeEach(() => {
  calls.length = 0
  queues = {
    'products.select': [
      { data: PRODUCTS, error: null },      // ครั้งที่ 1: รายการที่จะส่งออก
      { data: EXISTING_SKUS, error: null }, // ครั้งที่ 2: sku ที่ใช้ไปแล้ว
    ],
    'products.update': [{ data: null, error: null }],
    'taxonomy_nodes.select': [{ data: NODES, error: null }],
  }
})

describe('GET /api/pos-export', () => {
  it('คัดด้วย source + dedup_bucket + ยังไม่เคยส่งออก ไม่ใช่ status', async () => {
    const { GET } = await import('@/app/api/pos-export/route')
    await GET(new NextRequest('http://127.0.0.1:3000/api/pos-export'))

    const productCall = calls.find((c) => c.table === 'products')
    expect(productCall).toBeDefined()

    const eqFilters = productCall!.filters.filter((f) => f.kind === 'eq')
    expect(eqFilters).toEqual(
      expect.arrayContaining([
        { kind: 'eq', column: "metadata->>source", value: 'import_wizard' },
        { kind: 'eq', column: "metadata->>dedup_bucket", value: 'new' },
      ])
    )
    expect(productCall!.filters).toContainEqual({
      kind: 'is', column: 'pos_exported_at', value: null,
    })
    // status พลิกเป็น approved ตอน verify จึงใช้กรองไม่ได้
    expect(eqFilters.some((f) => f.column === 'status')).toBe(false)
  })

  it('คืนยอดรวมและจำนวนช่องบังคับที่ยังขาด', async () => {
    const { GET } = await import('@/app/api/pos-export/route')
    const response = await GET(new NextRequest('http://127.0.0.1:3000/api/pos-export'))
    const body = await response.json()

    expect(body.success).toBe(true)
    expect(body.total).toBe(2)
    // รหัสถูกออกให้แล้วตอน preview หน่วยตกเป็น ชิ้น เหลือแค่ราคาที่ยังไม่มี
    expect(body.issues).toEqual({
      missingItemId: 0, missingBarcode: 0, missingItemName: 0,
      missingUnitName: 0, missingItemGroup: 0, missingPrice: 1,
    })
  })

  it('หน้าตัวอย่างแสดงรหัสที่จะถูกออกให้ ไม่ใช่ช่องว่าง', async () => {
    const { GET } = await import('@/app/api/pos-export/route')
    const response = await GET(new NextRequest('http://127.0.0.1:3000/api/pos-export'))
    const body = await response.json()

    const ids = body.preview.map((r: { itemId: string }) => r.itemId)
    expect(ids).toContain('HH001')  // ของเดิมไม่ถูกแตะ
    expect(ids).toContain('ST134')  // ออกใหม่ เดินต่อจาก ST133
  })

  it('ตัวอย่างไม่เขียนอะไรลงฐานข้อมูล', async () => {
    const { GET } = await import('@/app/api/pos-export/route')
    await GET(new NextRequest('http://127.0.0.1:3000/api/pos-export'))

    expect(calls.some((c) => c.op === 'update')).toBe(false)
  })

  it('หน่วยที่ว่างตกเป็น ชิ้น', async () => {
    const { GET } = await import('@/app/api/pos-export/route')
    const response = await GET(new NextRequest('http://127.0.0.1:3000/api/pos-export'))
    const body = await response.json()

    expect(body.preview.map((r: { unitName: string }) => r.unitName)).toEqual(
      expect.arrayContaining(['ใบ', 'ชิ้น'])
    )
  })

  it('ตัวอย่างที่คืนมาแทน _ ในชื่อหมวดด้วยช่องว่างแล้ว', async () => {
    const { GET } = await import('@/app/api/pos-export/route')
    const response = await GET(new NextRequest('http://127.0.0.1:3000/api/pos-export'))
    const body = await response.json()

    expect(body.preview[0].itemGroup).toBe('เครื่องเขียน สำนักงาน')
    expect(body.preview[0].subGroup1).toBe('กาว/เทป')
  })
})

describe('middleware', () => {
  it('กัน GET /api/pos-export ไว้ เพราะอ่านด้วย service role', () => {
    const source = readFileSync(
      path.join(__dirname, '..', '..', 'middleware.ts'), 'utf8'
    )
    const gated = source.match(/GATED_READ_PATHS = new Set\(\[([^\]]*)\]/)
    expect(gated).not.toBeNull()
    expect(gated![1]).toContain('/api/pos-export')
  })
})
```

- [ ] **Step 2: รันเทสต์ให้เห็นว่าไม่ผ่าน**

Run: `npx jest __tests__/api/pos-export.test.ts`
Expected: FAIL — `Cannot find module '@/app/api/pos-export/route'`

- [ ] **Step 3: เขียน route**

สร้าง `taxonomy-app/app/api/pos-export/route.ts`:

```ts
import { NextRequest, NextResponse } from 'next/server'
import { supabaseAdmin } from '@/utils/supabase-admin'
import { withErrorHandling } from '@/utils/error-handler'
import {
  buildCategoryChain,
  countPosExportIssues,
  toPosExportRow,
  type ExportableProduct,
  type PosExportRow,
  type TaxonomyNodeLite,
} from '@/utils/pos-export'
import { allocateItemIds } from '@/utils/pos-item-id'

const PREVIEW_LIMIT = 50

/** จำนวนแถวที่ส่งกลับให้ UI ดูก่อนตัดสินใจโหลด — ไม่ใช่จำนวนที่จะอยู่ในไฟล์จริง */
export const dynamic = 'force-dynamic'

interface ExportProductRow extends ExportableProduct {
  id: string
}

/**
 * ดึงสินค้าที่ยังไม่ได้ส่งเข้า POS พร้อมแปลงเป็นแถวของไฟล์
 *
 * คัดด้วย metadata ไม่ใช่ status เพราะ /api/verify เซ็ต status เป็น 'approved'
 * ตอนคนกดยืนยันหมวด ซึ่งเป็นค่าเดียวกับสตอกเดิม 3,103 รายการ
 * metadata.source เป็นตัวเดียวที่ยังแยก 'import_wizard' ออกจาก 'legacy_labelled' ได้
 */
async function loadExportRows(): Promise<{
  ids: string[]
  rows: PosExportRow[]
  assignedSkus: Map<string, string>
}> {
  const { data: products, error: productsError } = await supabaseAdmin
    .from('products')
    .select('id, name_th, sku, barcode, unit_name, vat_flag, price, category_id')
    .eq('metadata->>source', 'import_wizard')
    .eq('metadata->>dedup_bucket', 'new')
    .is('pos_exported_at', null)
    .order('name_th')

  if (productsError) throw productsError

  // taxonomy มีแค่ 134 โหนด ดึงมาทั้งก้อนแล้วไล่สายในหน่วยความจำ
  // ถูกกว่ายิงหา parent ทีละรายการ และเลี่ยง raw SQL ตามกฎรีโป
  const { data: nodes, error: nodesError } = await supabaseAdmin
    .from('taxonomy_nodes')
    .select('id, name_th, parent_id')

  if (nodesError) throw nodesError

  // ต้องรู้ sku ทั้งหมดที่ใช้ไปแล้ว ไม่ใช่แค่เลขสูงสุด — products.sku มี UNIQUE
  // constraint ถ้าเลขในฐานข้อมูลขาดช่วง การเดินจาก max อย่างเดียวยังชนได้
  const { data: usedSkus, error: skuError } = await supabaseAdmin
    .from('products')
    .select('sku')
    .not('sku', 'is', null)

  if (skuError) throw skuError

  const nodesById = new Map<string, TaxonomyNodeLite>(
    ((nodes ?? []) as TaxonomyNodeLite[]).map((node) => [node.id, node])
  )

  const list = (products ?? []) as ExportProductRow[]

  // แจกรหัสทั้งใน GET และ POST ด้วยข้อมูลชุดเดียวกัน ต่างกันแค่ POST เขียนลงฐานข้อมูล
  // ตัวแจกเรียงตามชื่อเสมอ รหัสที่เห็นในหน้าตัวอย่างจึงตรงกับไฟล์ที่โหลดจริง
  const assignedSkus = allocateItemIds(
    list.map((product) => ({
      id: product.id,
      name_th: product.name_th,
      sku: product.sku,
      mainCategory: buildCategoryChain(product.category_id, nodesById)[0] ?? null,
    })),
    ((usedSkus ?? []) as { sku: string | null }[])
      .map((row) => row.sku ?? '')
      .filter(Boolean)
  )

  return {
    ids: list.map((product) => product.id),
    rows: list.map((product) =>
      toPosExportRow(
        { ...product, sku: product.sku ?? assignedSkus.get(product.id) ?? null },
        nodesById
      )
    ),
    assignedSkus,
  }
}

export { loadExportRows }

/**
 * GET /api/pos-export
 * ยอดรวม + จำนวนช่องที่ยังขาด + ตัวอย่างแถวแรก ๆ ให้ดูก่อนกดโหลด
 */
export async function GET(_request: NextRequest): Promise<NextResponse> {
  return withErrorHandling(async () => {
    const { rows } = await loadExportRows()

    return NextResponse.json({
      success: true,
      total: rows.length,
      issues: countPosExportIssues(rows),
      preview: rows.slice(0, PREVIEW_LIMIT),
    })
  })
}
```

- [ ] **Step 4: เพิ่ม path เข้า middleware**

แก้ `taxonomy-app/middleware.ts` บรรทัด 18:

```ts
const GATED_READ_PATHS = new Set(['/api/settings', '/api/import/history', '/api/pos-export'])
```

- [ ] **Step 5: รันเทสต์ให้ผ่าน**

Run: `npx jest __tests__/api/pos-export.test.ts`
Expected: PASS ทุกเคส รวมเคส middleware

- [ ] **Step 6: Commit**

```bash
git add taxonomy-app/app/api/pos-export/route.ts taxonomy-app/middleware.ts taxonomy-app/__tests__/api/pos-export.test.ts
git commit -m "feat: add GET /api/pos-export preview endpoint

คัดด้วย metadata.source ไม่ใช่ status เพราะ verify พลิก status เป็น approved
ปนกับสตอกเดิม 3,103 รายการ และเพิ่ม path เข้า GATED_READ_PATHS
เพราะ route อ่านด้วย service role ซึ่งข้าม RLS

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: API route — ดาวน์โหลดไฟล์และทำเครื่องหมายว่าส่งออกแล้ว (POST)

**Files:**
- Modify: `taxonomy-app/app/api/pos-export/route.ts` — เพิ่ม `POST`
- Modify: `taxonomy-app/__tests__/api/pos-export.test.ts` — เพิ่ม describe block

**Interfaces:**
- Consumes: `loadExportRows` จาก Task 4 · `buildPosExportFile` จาก Task 1 · `allocateItemIds` จาก Task 3
- Produces: `POST /api/pos-export` → body เป็นข้อความไฟล์ `text/plain; charset=utf-8` พร้อม `Content-Disposition: attachment`

- [ ] **Step 1: เขียนเทสต์ที่ยังไม่ผ่าน**

เพิ่มท้าย `taxonomy-app/__tests__/api/pos-export.test.ts`:

```ts
describe('POST /api/pos-export', () => {
  const post = async (body: unknown) => {
    const { POST } = await import('@/app/api/pos-export/route')
    return POST(
      new NextRequest('http://127.0.0.1:3000/api/pos-export', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body),
      })
    )
  }

  it('คืนไฟล์เป็นข้อความ ไม่ใช่ JSON', async () => {
    const response = await post({})
    expect(response.headers.get('content-type')).toContain('text/plain')
    expect(response.headers.get('content-type')).toContain('charset=utf-8')
    expect(response.headers.get('content-disposition')).toContain('attachment')
  })

  it('เนื้อไฟล์มีหัวตารางและครบทุกแถว', async () => {
    const response = await post({})
    const text = await response.text()
    const lines = text.split('\r\n').filter(Boolean)

    expect(lines).toHaveLength(3) // หัวตาราง + 2 สินค้า
    expect(lines[0].split('\t')).toHaveLength(9)
    expect(text.endsWith('\r\n')).toBe(true)
    expect(text.charCodeAt(0)).not.toBe(0xfeff)
  })

  it('เขียนรหัสที่ออกใหม่ลง products.sku', async () => {
    await post({})

    const skuUpdate = calls.find(
      (c) => c.op === 'update' && (c.payload as Record<string, unknown>)?.sku !== undefined
    )
    expect(skuUpdate).toBeDefined()
    expect(skuUpdate!.payload).toEqual({ sku: 'ST134' })
    expect(skuUpdate!.filters).toContainEqual({ kind: 'eq', column: 'id', value: 'p2' })
  })

  it('ไม่เขียนทับรหัสของรายการที่มีอยู่แล้ว', async () => {
    await post({})

    const skuUpdates = calls.filter(
      (c) => c.op === 'update' && (c.payload as Record<string, unknown>)?.sku !== undefined
    )
    expect(skuUpdates).toHaveLength(1) // p1 มี HH001 อยู่แล้ว ไม่ถูกแตะ
  })

  it('รหัสในไฟล์ตรงกับรหัสที่เขียนลงฐานข้อมูล', async () => {
    const response = await post({})
    const text = await response.text()

    expect(text).toContain('ST134')
  })

  it('ทำเครื่องหมาย pos_exported_at ให้แถวที่ส่งออกไป', async () => {
    await post({})

    const marked = calls.find(
      (c) => c.op === 'update' && (c.payload as Record<string, unknown>)?.pos_exported_at !== undefined
    )
    expect(marked).toBeDefined()
    expect(marked!.filters).toContainEqual({
      kind: 'in', column: 'id', value: ['p1', 'p2'],
    })
  })

  it('เขียนรหัสก่อนแล้วค่อยทำเครื่องหมายส่งออก', async () => {
    await post({})

    const updates = calls.filter((c) => c.op === 'update')
    const skuAt = updates.findIndex((c) => (c.payload as Record<string, unknown>)?.sku !== undefined)
    const markAt = updates.findIndex(
      (c) => (c.payload as Record<string, unknown>)?.pos_exported_at !== undefined
    )
    // ถ้าเขียนรหัสพลาด ต้องยังไม่มีอะไรถูกตีว่าส่งออกแล้ว ของจะได้ค้างในคิว
    expect(skuAt).toBeLessThan(markAt)
  })

  it('mark_exported=false โหลดไฟล์ได้โดยไม่แตะฐานข้อมูล', async () => {
    await post({ mark_exported: false })

    expect(calls.some((c) => c.op === 'update')).toBe(false)
  })

  it('ไม่มีอะไรให้ส่งออกก็ตอบ 409 พร้อมข้อความไทย ไม่ใช่ไฟล์เปล่า', async () => {
    results.products = { data: [], error: null }

    const response = await post({})
    expect(response.status).toBe(409)

    const body = await response.json()
    expect(body.success).toBe(false)
    expect(body.error).toContain('ไม่มีสินค้า')
  })
})
```

- [ ] **Step 2: รันเทสต์ให้เห็นว่าไม่ผ่าน**

Run: `npx jest __tests__/api/pos-export.test.ts -t "POST"`
Expected: FAIL — `POST is not a function`

- [ ] **Step 3: เขียน POST**

เพิ่มใน `taxonomy-app/app/api/pos-export/route.ts` — เพิ่ม import และต่อท้ายไฟล์:

```ts
import { z } from 'zod'
import { buildPosExportFile } from '@/utils/pos-export'

const ExportRequestSchema = z.object({
  // ปิดไว้เพื่อลองโหลดดูก่อนโดยไม่กินโควตา "ส่งออกแล้ว"
  mark_exported: z.boolean().optional().default(true),
})

/** จำนวน update ที่ยิงพร้อมกันตอนเขียนรหัสใหม่ */
const SKU_UPDATE_BATCH = 25

/** ชื่อไฟล์ตามวันที่ เพื่อให้เก็บหลายรอบไว้ในโฟลเดอร์เดียวกันได้ */
function exportFileName(): string {
  const today = new Date().toISOString().slice(0, 10)
  return `pos-import-${today}.txt`
}

/**
 * POST /api/pos-export
 * สร้างไฟล์ แล้วทำเครื่องหมายว่าสินค้าชุดนี้ส่งเข้า POS ไปแล้ว
 *
 * เป็น POST ไม่ใช่ GET เพราะเขียนฐานข้อมูล และ middleware กัน non-GET ไว้อยู่แล้ว
 */
export async function POST(request: NextRequest): Promise<NextResponse> {
  return withErrorHandling(async () => {
    const raw = await request.json().catch(() => ({}))
    const { mark_exported: markExported } = ExportRequestSchema.parse(raw)

    const { ids, rows, assignedSkus } = await loadExportRows()

    if (rows.length === 0) {
      return NextResponse.json(
        { success: false, error: 'ไม่มีสินค้าที่รอส่งเข้า POS' },
        { status: 409 }
      )
    }

    const fileText = buildPosExportFile(rows)

    // เขียนฐานข้อมูลหลังสร้างไฟล์สำเร็จเท่านั้น
    // ถ้า build พังก่อนหน้านี้ ของจะยังค้างรอส่งออกอยู่ ไม่หายไปเงียบ ๆ
    if (markExported) {
      // แต่ละแถวได้คนละรหัส อัปเดตรวดเดียวด้วย .in() ไม่ได้
      // ยิงทีละก้อนเพื่อไม่ให้เปิดคอนเนกชันพร้อมกันเป็นร้อย
      //
      // ถ้าพังกลางคัน บางรายการจะมีรหัสแล้วแต่ยังไม่ถูกตีว่าส่งออก — กดใหม่ได้
      // ตัวแจกรหัสข้ามรายการที่มี sku แล้ว รหัสเดิมจึงไม่ถูกเปลี่ยน
      const entries = [...assignedSkus.entries()]
      for (let start = 0; start < entries.length; start += SKU_UPDATE_BATCH) {
        const results = await Promise.all(
          entries.slice(start, start + SKU_UPDATE_BATCH).map(([productId, sku]) =>
            supabaseAdmin.from('products').update({ sku }).eq('id', productId)
          )
        )
        const failed = results.find((result) => result.error)
        if (failed?.error) throw failed.error
      }

      // ทำเครื่องหมายเป็นขั้นสุดท้าย — ต้องแน่ใจก่อนว่าทุกแถวมีรหัสแล้ว
      const { error } = await supabaseAdmin
        .from('products')
        .update({ pos_exported_at: new Date().toISOString() })
        .in('id', ids)

      if (error) throw error
    }

    return new NextResponse(fileText, {
      status: 200,
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'Content-Disposition': `attachment; filename="${exportFileName()}"`,
        'Cache-Control': 'no-store',
      },
    })
  })
}
```

- [ ] **Step 4: รันเทสต์ให้ผ่าน**

Run: `npx jest __tests__/api/pos-export.test.ts`
Expected: PASS ทุกเคสทั้ง GET และ POST

- [ ] **Step 5: ลองยิงจริงกับฐานข้อมูลจริง**

เปิด dev server (`npm run dev`) แล้วรันจากรากรีโป — ต้องมี `INTERNAL_API_SECRET` ใน `taxonomy-app/.env.local` ก่อน:

```bash
SECRET=$(grep -oP '(?<=^INTERNAL_API_SECRET=).*' taxonomy-app/.env.local | tr -d '\r')
TOKEN=$(node -e "console.log(require('crypto').createHash('sha256').update(process.argv[1]).digest('hex'))" "$SECRET")
curl -s -b "internal_session=$TOKEN" http://localhost:3000/api/pos-export | head -c 400
```

Expected: `"total":221` และ issues ที่ **`missingPrice` เป็น 221 ส่วนอีก 5 ช่องเป็น 0**
— รหัสสินค้าถูกออกให้แล้ว หน่วยตกเป็น `ชิ้น` เหลือแค่ราคาที่ยังไม่มีข้อมูล ตามที่ตกลงกันไว้

ตรวจด้วยว่า `preview[0].itemId` ขึ้นต้นด้วย prefix ที่ถูกต้อง (เช่น `HH696`) ไม่ใช่ค่าว่าง
และ **ยิง GET ซ้ำอีกครั้งต้องได้รหัสชุดเดิม** — ถ้าได้คนละชุด แปลว่าตัวแจกไม่คงเส้นคงวา
หน้าจอกับไฟล์จริงจะไม่ตรงกัน ต้องแก้ก่อนไปต่อ

- [ ] **Step 6: Commit**

```bash
git add taxonomy-app/app/api/pos-export/route.ts taxonomy-app/__tests__/api/pos-export.test.ts
git commit -m "feat: add POST /api/pos-export download endpoint

ทำเครื่องหมาย pos_exported_at หลังสร้างไฟล์สำเร็จเท่านั้น
เพื่อไม่ให้ของหายจากคิวเมื่อสร้างไฟล์พลาด

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: แท็บ UI ในหน้า /data-quality

**Files:**
- Create: `taxonomy-app/components/data-quality/PosExportTab.tsx`
- Modify: `taxonomy-app/app/data-quality/page.tsx:13` (union ของ `activeTab`), `:63-73` (ปุ่มแท็บ), `:76-87` (ส่วนเนื้อหา)
- Test: `taxonomy-app/__tests__/components/data-quality/pos-export-tab.test.tsx`

**Interfaces:**
- Consumes: `GET /api/pos-export` และ `POST /api/pos-export` จาก Task 4-5 · `PosExportRow`, `PosExportIssues` จาก Task 1
- Produces: `data-testid` ที่ Task 7 ใช้ — `pos-export-tab`, `tab-pos-export`, `pos-export-total`, `pos-export-warning`, `pos-export-download`, `pos-export-empty`, `pos-export-loading`, `pos-export-row-<index>`

- [ ] **Step 1: เขียนเทสต์ที่ยังไม่ผ่าน**

สร้าง `taxonomy-app/__tests__/components/data-quality/pos-export-tab.test.tsx`:

```tsx
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import PosExportTab from '@/components/data-quality/PosExportTab'

const previewRow = {
  itemGroup: 'ของใช้ในบ้าน',
  subGroup1: 'ตะกร้า/กระจาด',
  subGroup2: '',
  itemId: 'HH001',
  barcode: 'HH001',
  itemName: 'ตะกร้ากลม 9899',
  unitName: 'ใบ',
  price: 45,
  vat: 'V',
}

const EMPTY_ISSUES = {
  missingItemId: 0, missingBarcode: 0, missingItemName: 0,
  missingUnitName: 0, missingItemGroup: 0, missingPrice: 0,
}

const mockGet = (body: unknown) => {
  global.fetch = jest.fn().mockResolvedValue({
    ok: true,
    json: async () => body,
  }) as unknown as typeof fetch
}

afterEach(() => jest.resetAllMocks())

it('แสดงยอดรวมที่รอส่งออก', async () => {
  mockGet({ success: true, total: 221, issues: EMPTY_ISSUES, preview: [previewRow] })
  render(<PosExportTab />)

  await waitFor(() =>
    expect(screen.getByTestId('pos-export-total')).toHaveTextContent('221')
  )
})

it('เตือนเฉพาะช่องบังคับที่ยังขาดจริง', async () => {
  mockGet({
    success: true,
    total: 221,
    issues: { ...EMPTY_ISSUES, missingPrice: 221 },
    preview: [previewRow],
  })
  render(<PosExportTab />)

  const warning = await screen.findByTestId('pos-export-warning')
  expect(warning).toHaveTextContent('221')
  expect(warning).toHaveTextContent('ราคาขาย')
  // รหัสสินค้าถูกออกให้อัตโนมัติแล้ว ไม่ควรขึ้นเตือน
  expect(warning).not.toHaveTextContent('รหัสสินค้า')
})

it('เตือนได้หลายช่องพร้อมกัน', async () => {
  mockGet({
    success: true,
    total: 10,
    issues: { ...EMPTY_ISSUES, missingPrice: 10, missingUnitName: 3 },
    preview: [previewRow],
  })
  render(<PosExportTab />)

  const warning = await screen.findByTestId('pos-export-warning')
  expect(warning).toHaveTextContent('ราคาขาย')
  expect(warning).toHaveTextContent('ชื่อหน่วย')
})

it('ไม่เตือนเมื่อข้อมูลครบ', async () => {
  mockGet({ success: true, total: 5, issues: EMPTY_ISSUES, preview: [previewRow] })
  render(<PosExportTab />)

  await waitFor(() => expect(screen.getByTestId('pos-export-total')).toBeInTheDocument())
  expect(screen.queryByTestId('pos-export-warning')).not.toBeInTheDocument()
})

it('ปิดปุ่มโหลดเมื่อไม่มีอะไรให้ส่งออก', async () => {
  mockGet({ success: true, total: 0, issues: EMPTY_ISSUES, preview: [] })
  render(<PosExportTab />)

  expect(await screen.findByTestId('pos-export-empty')).toBeInTheDocument()
  expect(screen.getByTestId('pos-export-download')).toBeDisabled()
})

it('กดโหลดแล้วยิง POST ไปที่ /api/pos-export', async () => {
  const fetchMock = jest.fn()
    .mockResolvedValueOnce({
      ok: true,
      json: async () => ({ success: true, total: 1, issues: EMPTY_ISSUES, preview: [previewRow] }),
    })
    .mockResolvedValueOnce({
      ok: true,
      blob: async () => new Blob(['header\r\nrow\r\n'], { type: 'text/plain' }),
    })
  global.fetch = fetchMock as unknown as typeof fetch
  global.URL.createObjectURL = jest.fn(() => 'blob:fake')
  global.URL.revokeObjectURL = jest.fn()

  render(<PosExportTab />)
  await waitFor(() => expect(screen.getByTestId('pos-export-total')).toBeInTheDocument())

  await userEvent.click(screen.getByTestId('pos-export-download'))

  await waitFor(() => {
    expect(fetchMock).toHaveBeenLastCalledWith(
      '/api/pos-export',
      expect.objectContaining({ method: 'POST' })
    )
  })
})
```

- [ ] **Step 2: รันเทสต์ให้เห็นว่าไม่ผ่าน**

Run: `npx jest __tests__/components/data-quality/pos-export-tab.test.tsx`
Expected: FAIL — `Cannot find module '@/components/data-quality/PosExportTab'`

- [ ] **Step 3: เขียนคอมโพเนนต์**

สร้าง `taxonomy-app/components/data-quality/PosExportTab.tsx`:

```tsx
'use client'

import { useCallback, useEffect, useState } from 'react'
import { AlertTriangle, Download, RefreshCcw } from 'lucide-react'
import type { PosExportIssues, PosExportRow } from '@/utils/pos-export'

interface ExportPreview {
  total: number
  issues: PosExportIssues
  preview: PosExportRow[]
}

/** ช่องบังคับทั้ง 6 ของ POS — ขาดช่องไหน POS ไม่รับไฟล์ */
const MANDATORY_FIELDS: { key: keyof PosExportIssues; label: string }[] = [
  { key: 'missingItemId', label: 'รหัสสินค้า' },
  { key: 'missingBarcode', label: 'บาร์โค้ด' },
  { key: 'missingItemName', label: 'ชื่อสินค้า' },
  { key: 'missingUnitName', label: 'ชื่อหน่วย' },
  { key: 'missingItemGroup', label: 'กลุ่มสินค้า' },
  { key: 'missingPrice', label: 'ราคาขาย' },
]

const NO_ISSUES: PosExportIssues = {
  missingItemId: 0, missingBarcode: 0, missingItemName: 0,
  missingUnitName: 0, missingItemGroup: 0, missingPrice: 0,
}

/**
 * ส่งออกสินค้าใหม่ที่ยังไม่มีในสตอก เป็นไฟล์ .txt ที่ POS นำเข้าได้
 *
 * เตือนจำนวนแถวที่ยังไม่มีรหัสสินค้า/ราคาก่อนโหลด เพราะสองช่องนั้น
 * ยังว่างทั้งตารางจนกว่าจะ map คอลัมน์ตอน import รอบหน้า
 * (ดู docs/specs/2026-09-06-pos-export.md หัวข้อข้อจำกัด)
 */
export default function PosExportTab() {
  const [data, setData] = useState<ExportPreview | null>(null)
  const [loading, setLoading] = useState(true)
  const [downloading, setDownloading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchPreview = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch('/api/pos-export')
      const payload = await response.json()
      if (!payload.success) throw new Error(payload.error || 'โหลดข้อมูลไม่สำเร็จ')
      setData({ total: payload.total, issues: payload.issues, preview: payload.preview })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'โหลดข้อมูลไม่สำเร็จ')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void fetchPreview()
  }, [fetchPreview])

  const handleDownload = useCallback(async () => {
    setDownloading(true)
    setError(null)
    try {
      const response = await fetch('/api/pos-export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mark_exported: true }),
      })

      if (!response.ok) {
        const payload = await response.json().catch(() => null)
        throw new Error(payload?.error || 'สร้างไฟล์ไม่สำเร็จ')
      }

      // ต้องผ่าน blob เพราะ response มาจาก POST ที่พาคุกกี้ไปด้วย
      // จะใช้ <a href> ตรง ๆ ไม่ได้ — นั่นเป็น GET
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `pos-import-${new Date().toISOString().slice(0, 10)}.txt`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      URL.revokeObjectURL(url)

      await fetchPreview()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'สร้างไฟล์ไม่สำเร็จ')
    } finally {
      setDownloading(false)
    }
  }, [fetchPreview])

  if (loading) {
    return (
      <div data-testid="pos-export-loading" className="p-8 text-center text-slate-400">
        กำลังโหลด...
      </div>
    )
  }

  const total = data?.total ?? 0
  const issues = data?.issues ?? NO_ISSUES
  const gaps = MANDATORY_FIELDS.filter(({ key }) => issues[key] > 0)

  return (
    <div data-testid="pos-export-tab" className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-black text-slate-800">ส่งออกเข้า POS</h2>
          <p className="text-sm text-slate-500">
            สินค้าใหม่ที่ยังไม่มีในสตอก และยังไม่เคยส่งออก
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => void fetchPreview()}
            className="flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-bold text-slate-500 hover:bg-slate-100"
          >
            <RefreshCcw className="h-4 w-4" /> รีเฟรช
          </button>
          <button
            data-testid="pos-export-download"
            onClick={() => void handleDownload()}
            disabled={total === 0 || downloading}
            className="flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-bold text-white shadow-sm hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            <Download className="h-4 w-4" />
            {downloading ? 'กำลังสร้างไฟล์...' : 'ดาวน์โหลด .txt'}
          </button>
        </div>
      </div>

      <p data-testid="pos-export-total" className="text-sm font-bold text-slate-600">
        รอส่งออก <span className="text-2xl font-black text-indigo-600">{total}</span> รายการ
      </p>

      {error && (
        <p className="rounded-lg bg-rose-50 p-3 text-sm font-bold text-rose-600">{error}</p>
      )}

      {gaps.length > 0 && (
        <div
          data-testid="pos-export-warning"
          className="flex gap-3 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800"
        >
          <AlertTriangle className="h-5 w-5 shrink-0" />
          <div className="space-y-1">
            <p className="font-black">POS บังคับช่องเหล่านี้ แต่ยังว่างอยู่</p>
            {gaps.map(({ key, label }) => (
              <p key={key}>
                ยังไม่มี<strong>{label}</strong> {issues[key]} รายการ
              </p>
            ))}
            <p className="text-xs">
              เติมได้ตอนนำเข้ารอบหน้า โดยเลือกคอลัมน์ที่ตรงกันในขั้นจับคู่คอลัมน์
              — ดาวน์โหลดตอนนี้ได้ แต่ POS อาจไม่รับไฟล์
            </p>
          </div>
        </div>
      )}

      {total === 0 ? (
        <p data-testid="pos-export-empty" className="rounded-lg bg-slate-50 p-8 text-center text-sm font-bold text-slate-400">
          ไม่มีสินค้าที่รอส่งเข้า POS
        </p>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-slate-200">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 text-[10px] font-black uppercase text-slate-400">
              <tr>
                <th className="whitespace-nowrap px-4 py-3">กลุ่มสินค้า</th>
                <th className="whitespace-nowrap px-4 py-3">กลุ่มย่อย 1</th>
                <th className="whitespace-nowrap px-4 py-3">รหัสสินค้า</th>
                <th className="whitespace-nowrap px-4 py-3">ชื่อสินค้า</th>
                <th className="whitespace-nowrap px-4 py-3">หน่วย</th>
                <th className="whitespace-nowrap px-4 py-3">ราคา</th>
              </tr>
            </thead>
            <tbody>
              {(data?.preview ?? []).map((row, index) => (
                <tr
                  key={`${row.itemName}-${index}`}
                  data-testid={`pos-export-row-${index}`}
                  className="border-t border-slate-100"
                >
                  <td className="px-4 py-3 text-slate-500">{row.itemGroup}</td>
                  <td className="px-4 py-3 text-slate-500">{row.subGroup1}</td>
                  <td className="px-4 py-3 font-bold text-slate-700">
                    {row.itemId || <span className="text-rose-400">— ไม่มี —</span>}
                  </td>
                  <td className="max-w-[280px] truncate px-4 py-3 font-bold text-slate-700">
                    {row.itemName}
                  </td>
                  <td className="px-4 py-3 text-slate-500">{row.unitName || '—'}</td>
                  <td className="px-4 py-3 text-slate-500">
                    {row.price === null ? <span className="text-rose-400">—</span> : row.price}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 4: ต่อแท็บเข้าหน้า /data-quality**

แก้ `taxonomy-app/app/data-quality/page.tsx` สามจุด:

1. เพิ่ม import ต่อจากบรรทัด 9:

```tsx
import PosExportTab from '@/components/data-quality/PosExportTab'
```

และเพิ่ม `Upload` เข้า import ของ `lucide-react` บรรทัด 10

2. ขยาย union บรรทัด 13:

```tsx
const [activeTab, setActiveTab] = useState<'dedup' | 'verify' | 'recheck' | 'autolearn' | 'posexport'>('dedup')
```

3. เพิ่มปุ่มแท็บต่อจากปุ่ม autolearn (ราวบรรทัด 73) — ลอกโครงจากปุ่ม `tab-recheck`:

```tsx
              <button
                onClick={() => setActiveTab('posexport')}
                className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-bold transition-all ${
                  activeTab === 'posexport' ? 'bg-white text-indigo-600 shadow-md border border-slate-200/50' : 'text-slate-500 hover:text-slate-700 hover:bg-slate-200/50'
                }`}
                data-testid="tab-pos-export"
              >
                <Upload className="w-4 h-4" />
                ส่งออก POS
              </button>
```

4. เพิ่มส่วนเนื้อหาต่อจากบรรทัด 86:

```tsx
          {activeTab === 'posexport' && <PosExportTab />}
```

- [ ] **Step 5: รันเทสต์ให้ผ่าน**

Run: `npx jest __tests__/components/data-quality/pos-export-tab.test.tsx`
Expected: PASS ทุกเคส

- [ ] **Step 6: รันชุดเทสต์ทั้งหมด + typecheck**

Run: `npx jest --ci` แล้ว `npx tsc --noEmit`
Expected:
- jest — ต้องมากกว่า **195 passed** เดิม (เพิ่มไฟล์เทสต์ใหม่จาก Task 1/3/4/5/6) ขอแค่ **ไม่มี failed เพิ่ม** และยังคง 4 suite-level failures เดิม (`__tests__/integration/*`, `__tests__/setup/database-setup.ts`) ที่พังเพราะ jest ไม่โหลด `.env.local` — ไม่เกี่ยวกับงานนี้
- tsc — **2 errors เท่าเดิม** ใน `__tests__/integration/synonym.integration.test.ts`

ถ้าเลขพังเพิ่มจากนี้ ต้องแก้ก่อนไปต่อ

- [ ] **Step 7: Commit**

```bash
git add taxonomy-app/components/data-quality/PosExportTab.tsx taxonomy-app/app/data-quality/page.tsx taxonomy-app/__tests__/components/data-quality/pos-export-tab.test.tsx
git commit -m "feat: add POS export tab to /data-quality

เตือนจำนวนแถวที่ยังไม่มีรหัสสินค้า/ราคาก่อนโหลด
เพื่อไม่ให้ไฟล์ที่มีช่องว่างหลุดไปเข้า POS โดยไม่รู้ตัว

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: เทสต์ end-to-end กับฐานข้อมูลจริง

**Files:**
- Create: `taxonomy-app/e2e/pos-export.spec.ts`

**Interfaces:**
- Consumes: `data-testid` ทั้งหมดจาก Task 6

> **e2e ในรีโปนี้เขียนฐานข้อมูลจริง** สเปกนี้ตั้งใจ**ไม่**กดปุ่มดาวน์โหลดที่ `mark_exported: true` เพราะจะไปตั้ง `pos_exported_at` ให้สินค้าจริง 221 รายการ ทำให้ของหายจากคิวถาวร เทสต์จึงตรวจแค่การแสดงผลกับการยิง POST แบบ `mark_exported: false`

- [ ] **Step 1: เขียนสเปก**

สร้าง `taxonomy-app/e2e/pos-export.spec.ts`:

```ts
import { test, expect } from '@playwright/test'

/**
 * แท็บส่งออกเข้า POS (Data Quality → ส่งออก POS)
 *
 * ต้องมีสินค้าที่ metadata.source='import_wizard' และ dedup_bucket='new'
 * อยู่ในฐานข้อมูลก่อน ไม่งั้นจะเห็นแต่สถานะว่าง (ซึ่งก็ยังตรวจได้)
 *
 * สเปกนี้ไม่กดดาวน์โหลดจริง — POST ที่ mark_exported=true จะตั้ง
 * pos_exported_at ให้ของจริง ทำให้หายจากคิวถาวร
 */
test.describe('POS export', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/data-quality')
    await page.waitForLoadState('networkidle')
    await page.locator('[data-testid="tab-pos-export"]').click()

    await expect(page.locator('[data-testid="pos-export-loading"]')).toHaveCount(0)
    await expect(page.locator('[data-testid="pos-export-tab"]')).toBeVisible()
  })

  test('แสดงยอดรวมที่รอส่งออก', async ({ page }) => {
    await expect(page.locator('[data-testid="pos-export-total"]')).toContainText('รอส่งออก')
  })

  test('มีของก็แสดงตาราง ไม่มีก็แสดงสถานะว่างและปิดปุ่ม', async ({ page }) => {
    const empty = page.locator('[data-testid="pos-export-empty"]')
    const firstRow = page.locator('[data-testid="pos-export-row-0"]')

    await expect(empty.or(firstRow)).toBeVisible()

    if (await empty.isVisible()) {
      await expect(page.locator('[data-testid="pos-export-download"]')).toBeDisabled()
    } else {
      await expect(page.locator('[data-testid="pos-export-download"]')).toBeEnabled()
    }
  })

  test('ไฟล์ที่ API สร้างมีหัวตาราง 9 คอลัมน์ ขึ้นบรรทัดแบบ CRLF และไม่มี BOM', async ({ page, request }) => {
    const total = await page.locator('[data-testid="pos-export-total"]').textContent()
    test.skip(/\b0\b/.test(total ?? ''), 'ไม่มีสินค้ารอส่งออก')

    // mark_exported=false — ดูไฟล์ได้โดยไม่ทำให้ของหายจากคิว
    const response = await request.post('/api/pos-export', {
      data: { mark_exported: false },
    })
    expect(response.status()).toBe(200)
    expect(response.headers()['content-type']).toContain('text/plain')

    const text = await response.text()
    expect(text.charCodeAt(0)).not.toBe(0xfeff)
    expect(text.endsWith('\r\n')).toBe(true)

    const lines = text.split('\r\n').filter(Boolean)
    expect(lines[0].split('\t')).toHaveLength(9)

    // ไม้เอกซ้ำสองตัวในหัวคอลัมน์ที่ 5 ต้องรอดมาถึงปลายทาง
    expect(lines[0].split('\t')[4]).toBe(
      'Item Name(\u0E0A\u0E37\u0E48\u0E48\u0E2D\u0E2A\u0E34\u0E19\u0E04\u0E49\u0E32)'
    )

    for (const line of lines) {
      expect(line.split('\t')).toHaveLength(9)
    }
    expect(text).not.toContain('"')
  })

  test('ไม่มี horizontal scroll ที่จอ 375px', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 800 })
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth > document.documentElement.clientWidth
    )
    expect(overflow).toBe(false)
  })
})
```

- [ ] **Step 2: รันสเปกนี้**

Run: `npx playwright test e2e/pos-export.spec.ts`
Expected: 4 tests ผ่าน (บางตัวอาจ skip ถ้าไม่มีสินค้ารอส่งออก)

ต้องมี `INTERNAL_API_SECRET` ใน `taxonomy-app/.env.local` ก่อน ไม่งั้น `playwright.setup.ts` ปลดล็อกไม่ได้ และ POST จะได้ 401

- [ ] **Step 3: ยืนยันว่าไม่ทำสเปกเดิมพัง**

Run: `npx playwright test e2e/synonym-management.spec.ts e2e/recheck-legacy.spec.ts`
Expected: 9 tests ผ่านทั้งหมด (เท่าเดิม — สองไฟล์นี้คือชุดที่เขียวอยู่แล้ว)

- [ ] **Step 4: Commit**

```bash
git add taxonomy-app/e2e/pos-export.spec.ts
git commit -m "test: add e2e spec for POS export tab

ไม่กดดาวน์โหลดจริงเพราะจะตั้ง pos_exported_at ให้ของจริงถาวร
ใช้ mark_exported=false ตรวจโครงไฟล์แทน

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 5: บันทึกลง CURRENT_STATUS.md**

เพิ่ม session entry ท้าย `docs/CURRENT_STATUS.md` ตามรูปแบบที่ไฟล์นั้นใช้อยู่ ระบุ:
- เพิ่มการส่งออกไฟล์นำเข้า POS ที่ `/data-quality` → แท็บ "ส่งออก POS"
- คัดของด้วย `metadata.source` + `dedup_bucket` ไม่ใช่ `status` (พร้อมเหตุผล)
- 🔴 **ยังติด:** `sku` และ `price` ว่างทั้งตาราง ไฟล์ที่ได้จึงยังมี `Item ID`/`Price` ว่าง ต้อง map สองคอลัมน์นั้นตอน import รอบหน้า
- ตัวเลขเทสต์ที่วัดได้จริงหลังงานนี้

จากนั้น commit

---

## Self-Review

**ครอบคลุมสเปกครบ:**

| ข้อในสเปก | Task |
|---|---|
| หัวตาราง 9 คอลัมน์ + ไม้เอกซ้ำ | 1 (escape + เทสต์ codepoint), 7 (ตรวจปลายทาง) |
| tab-separated, UTF-8 ไม่มี BOM, CRLF, ไม่มีคำพูด | 1, 5, 7 |
| คงไว้ 9 คอลัมน์แม้ POS map ไม่ได้ 2 ช่อง | 1 (`POS_EXPORT_HEADERS`) |
| คัดด้วย source + dedup_bucket + ยังไม่ส่งออก | 4 (มีเทสต์ยืนยันว่าไม่แตะ `status`) |
| เพิ่มคอลัมน์จริงใน products | 2 |
| 1 แถว = 1 ชื่อสินค้า | ทั้งหมด — ไม่มีการแตกไซซ์ที่ไหนเลย |
| ราคาว่างได้ + จำนวนเต็มล้วน | 1 (`formatPrice`) |
| `_` → ช่องว่าง | 1 (`formatGroupName`) |
| Sub Group2 = taxonomy ชั้น 3 | 1 (`buildCategoryChain` + `toPosExportRow`) |
| barcode fallback ไป Item ID | 1 |
| **`Item ID` ออกอัตโนมัติตาม prefix หมวด** | 3 (ตัวแจก), 4 (ใช้ตอน preview), 5 (เขียนลง DB) |
| **`Unit Name` ว่างตกเป็น `ชิ้น`** | 1 (`DEFAULT_UNIT`) |
| **นับช่องบังคับครบทั้ง 6** | 1 (`countPosExportIssues`), 6 (UI แสดงเฉพาะที่ขาดจริง) |
| กันส่งออกซ้ำ | 2 (`pos_exported_at`), 5 (มาร์กเป็นขั้นสุดท้าย) |

**ชื่อและชนิดสอดคล้องกัน:** `PosExportRow`, `ExportableProduct`, `TaxonomyNodeLite`, `PosExportIssues`, `DEFAULT_UNIT` ประกาศครั้งเดียวใน Task 1 · `ItemIdCandidate`, `allocateItemIds`, `MAIN_CATEGORY_PREFIX` ประกาศครั้งเดียวใน Task 3 · `loadExportRows` export จาก Task 4 แล้ว Task 5 เรียกซ้ำ (คืน `assignedSkus` ที่ Task 5 ใช้เขียน DB) · `data-testid` ที่ Task 6 สร้างตรงกับที่ Task 7 ใช้ทุกตัว

**ความเสี่ยงที่รู้อยู่แล้ว:**
1. **`Price` จะยังว่างทั้ง 221 แถว** — ตกลงกันว่าไม่ backfill ไฟล์ที่ได้จึงยังขาดช่องบังคับหนึ่งช่อง UI เตือนไว้แล้วแต่ไม่บล็อกการโหลด
2. **รหัสที่แจกไปแล้วแก้ยาก** — พอเขียนลง `products.sku` และส่งเข้า POS แล้ว การเปลี่ยนทีหลังแปลว่าต้องแก้ทั้งสองฝั่ง Task 3 จึงมีเทสต์ล็อกทั้งความไม่ซ้ำและความคงเส้นคงวา
3. ตาราง `MAIN_CATEGORY_PREFIX` เป็นค่าคงที่ในโค้ด ถ้ามีหมวดหลักใหม่เพิ่มเข้ามาแล้วลืมเพิ่ม prefix สินค้าจะตกไปใช้ `MIS` เงียบ ๆ — Task 3 Step 5 ให้ตรวจกับข้อมูลจริงก่อน
4. `.eq('metadata->>source', ...)` ต้องยิงจริงดูสักครั้ง (Task 5 Step 5) — ถ้า supabase-js เวอร์ชันนี้ไม่รองรับ ต้องเปลี่ยนไปกรองในหน่วยความจำแทน
5. กดดาวน์โหลดจริงจะเขียน `sku` และตั้ง `pos_exported_at` ให้ 221 รายการ ถอยกลับได้ด้วย
   `UPDATE products SET pos_exported_at = NULL, sku = NULL WHERE metadata->>'source'='import_wizard' AND metadata->>'dedup_bucket'='new'`
