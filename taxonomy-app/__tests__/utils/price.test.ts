import { parsePrice, isPriceMismatch, classifyDedupBucket } from '@/utils/price'

describe('parsePrice', () => {
  it('parses plain numeric strings', () => {
    expect(parsePrice('79.0')).toBe(79)
    expect(parsePrice('100')).toBe(100)
  })

  it('strips thousand separators and currency symbols', () => {
    expect(parsePrice('1,234.50')).toBe(1234.5)
    expect(parsePrice('฿99')).toBe(99)
    expect(parsePrice(' 250 ')).toBe(250)
  })

  it('returns undefined for missing or invalid input', () => {
    expect(parsePrice(undefined)).toBeUndefined()
    expect(parsePrice(null)).toBeUndefined()
    expect(parsePrice('')).toBeUndefined()
    expect(parsePrice('-')).toBeUndefined()
    expect(parsePrice('abc')).toBeUndefined()
  })

  it('treats zero and negative as invalid (no real price is 0 in this catalog)', () => {
    expect(parsePrice('0')).toBeUndefined()
    expect(parsePrice('-50')).toBeUndefined()
  })

  it('accepts a raw number as-is', () => {
    expect(parsePrice(79)).toBe(79)
  })
})

describe('isPriceMismatch', () => {
  it('is false when prices are equal', () => {
    expect(isPriceMismatch(100, 100)).toBe(false)
  })

  it('is false when either price is missing', () => {
    expect(isPriceMismatch(undefined, 100)).toBe(false)
    expect(isPriceMismatch(100, undefined)).toBe(false)
    expect(isPriceMismatch(null, null)).toBe(false)
  })

  it('is false just inside the 2x boundary', () => {
    expect(isPriceMismatch(100, 199)).toBe(false)
  })

  it('is true right at and beyond the 2x boundary', () => {
    expect(isPriceMismatch(100, 200)).toBe(true)
    expect(isPriceMismatch(100, 300)).toBe(true)
  })

  it('is symmetric regardless of argument order', () => {
    expect(isPriceMismatch(200, 100)).toBe(true)
    expect(isPriceMismatch(100, 200)).toBe(true)
  })
})

describe('classifyDedupBucket', () => {
  // นโยบายปัจจุบัน: ทิ้งสินค้านำเข้าโดยไม่มีคนเห็นได้ก็ต่อเมื่อคะแนน ≥95% **และ**
  // กฎอ่านตัวเลขยืนยันว่าซ้ำ (ชื่อเหมือนเป๊ะหลังตัดเครื่องหมาย) — วัดแล้วคะแนนอย่างเดียว
  // ทิ้งผิด 3 จาก 37 รายการ เช่น "กะละมัง 004 58" ถูกจับคู่กับ "004/53"
  describe('auto-discard needs the identity rule to confirm', () => {
    it('discards ≥95% similarity when the rule confirms a duplicate and price is consistent', () => {
      expect(classifyDedupBucket(0.97, 'similar', 100, 110, 'duplicate')).toBe('duplicate')
    })

    it('sends ≥95% to review when the rule stays silent', () => {
      expect(classifyDedupBucket(0.97, 'similar', 100, 110, 'needs_judgment')).toBe('review')
    })

    it('sends ≥95% to review when the backend gave no verdict at all', () => {
      expect(classifyDedupBucket(0.97, 'similar', 100, 110)).toBe('review')
      expect(classifyDedupBucket(0.97, 'similar', 100, 110, undefined)).toBe('review')
    })

    it('demotes a confirmed duplicate to review when price differs by more than 2x', () => {
      expect(classifyDedupBucket(0.97, 'similar', 300, 100, 'duplicate')).toBe('review')
    })

    it('still discards a confirmed duplicate when price data is missing on either side', () => {
      expect(classifyDedupBucket(0.97, 'similar', undefined, 100, 'duplicate')).toBe('duplicate')
      expect(classifyDedupBucket(0.97, 'similar', 100, undefined, 'duplicate')).toBe('duplicate')
    })
  })

  // "ของใหม่" ยังมีคนยืนยันหมวดหมู่ต่ออีกด่าน ต่างจากการทิ้งที่ไม่มีใครเห็น
  // กฎที่ผิดจึงเสียหายน้อยกว่าฝั่งนั้นมาก
  describe('rule veto sends different products to new', () => {
    it('routes a ≥95% pair the rule calls different to new', () => {
      expect(classifyDedupBucket(0.97, 'similar', 100, 110, 'different')).toBe('new')
    })

    it('routes an 80-94% pair the rule calls different to new', () => {
      expect(classifyDedupBucket(0.85, 'similar', 100, 110, 'different')).toBe('new')
    })

    it('routes a below-80% pair the rule calls different to new', () => {
      expect(classifyDedupBucket(0.5, 'different', 100, 100, 'different')).toBe('new')
    })
  })

  describe('below the discard line, nothing else changes', () => {
    it('keeps the 80-94% band as review regardless of price when the rule is silent', () => {
      expect(classifyDedupBucket(0.85, 'different', 100, 110, 'needs_judgment')).toBe('review')
      expect(classifyDedupBucket(0.85, 'different', 300, 100, 'needs_judgment')).toBe('review')
    })

    it('keeps below-80% as new regardless of price', () => {
      expect(classifyDedupBucket(0.5, 'different', 100, 100)).toBe('new')
    })

    it('review zone via mlPrediction=similar below 80% is unaffected by price', () => {
      expect(classifyDedupBucket(0.78, 'similar', 300, 100, 'needs_judgment')).toBe('review')
    })
  })
})
