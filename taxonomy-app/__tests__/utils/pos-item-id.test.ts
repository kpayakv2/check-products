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

  it('ชื่อซ้ำใจเดียวกัน ต้องใช้ id เป็น tie-breaker เพื่อให้ผลตรงกันไม่ว่าลำดับ input อย่างไร', () => {
    const sameName = 'สินค้าเดียวกัน'
    const listA = [
      candidate({ id: 'z', name_th: sameName }),
      candidate({ id: 'a', name_th: sameName }),
      candidate({ id: 'm', name_th: sameName }),
    ]
    const listB = [
      candidate({ id: 'a', name_th: sameName }),
      candidate({ id: 'm', name_th: sameName }),
      candidate({ id: 'z', name_th: sameName }),
    ]
    const listC = [
      candidate({ id: 'm', name_th: sameName }),
      candidate({ id: 'z', name_th: sameName }),
      candidate({ id: 'a', name_th: sameName }),
    ]

    const assignedA = allocateItemIds(listA, ['HH695'])
    const assignedB = allocateItemIds(listB, ['HH695'])
    const assignedC = allocateItemIds(listC, ['HH695'])

    // ต้องได้รหัสตรงกันทั้งสามแบบ แม้ input array มีลำดับต่างกัน
    expect([...assignedA.entries()].sort()).toEqual([...assignedB.entries()].sort())
    expect([...assignedB.entries()].sort()).toEqual([...assignedC.entries()].sort())

    // และสั่งตามอักษร a < m < z
    expect(assignedA.get('a')).toBe('HH696')
    expect(assignedA.get('m')).toBe('HH697')
    expect(assignedA.get('z')).toBe('HH698')
  })
})
