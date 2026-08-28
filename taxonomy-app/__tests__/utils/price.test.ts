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
  it('auto-merges ≥95% similarity when price is consistent', () => {
    expect(classifyDedupBucket(0.97, 'similar', 100, 110)).toBe('duplicate')
  })

  it('demotes ≥95% similarity to review when price differs by more than 2x', () => {
    expect(classifyDedupBucket(0.97, 'similar', 300, 100)).toBe('review')
  })

  it('still auto-merges ≥95% similarity when price data is missing on either side', () => {
    expect(classifyDedupBucket(0.97, 'similar', undefined, 100)).toBe('duplicate')
    expect(classifyDedupBucket(0.97, 'similar', 100, undefined)).toBe('duplicate')
  })

  it('keeps the 80-94% band as review regardless of price', () => {
    expect(classifyDedupBucket(0.85, 'different', 100, 110)).toBe('review')
    expect(classifyDedupBucket(0.85, 'different', 300, 100)).toBe('review')
  })

  it('keeps below-80% as new regardless of price', () => {
    expect(classifyDedupBucket(0.5, 'different', 100, 100)).toBe('new')
  })

  it('review zone via mlPrediction=similar below 80% is unaffected by price', () => {
    expect(classifyDedupBucket(0.78, 'similar', 300, 100)).toBe('review')
  })
})
