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

