import {
  parseCSV,
  validateCSV,
  getColumnStats,
  detectEncoding
} from '@/utils/csv-parser'

describe('CSV Parser Utils', () => {
  describe('parseCSV', () => {
    it('should parse simple CSV', () => {
      const csv = `product_name,category,price
กล่องล็อค 560,unique,100
กล่องล็อค 561,unique,150`

      const result = parseCSV(csv)

      expect(result.headers).toEqual(['product_name', 'category', 'price'])
      expect(result.rows).toHaveLength(2)
      expect(result.totalCount).toBe(2)
      expect(result.rows[0]).toEqual({
        product_name: 'กล่องล็อค 560',
        category: 'unique',
        price: '100'
      })
    })

    it('should handle quoted values', () => {
      const csv = `name,description
"Product, with comma","Description ""quoted"""
Simple,Normal`

      const result = parseCSV(csv)

      expect(result.rows[0].name).toBe('Product, with comma')
      expect(result.rows[0].description).toBe('Description "quoted"')
    })

    it('should limit rows with maxRows option', () => {
      const csv = `name
Product 1
Product 2
Product 3
Product 4`

      const result = parseCSV(csv, { maxRows: 2 })

      expect(result.rows).toHaveLength(2)
      expect(result.totalCount).toBe(4)
    })

    it('should skip empty lines', () => {
      const csv = `name

Product 1

Product 2

`

      const result = parseCSV(csv, { skipEmptyLines: true })

      expect(result.rows).toHaveLength(2)
    })

    it('should handle empty CSV', () => {
      const result = parseCSV('')

      expect(result.headers).toEqual([])
      expect(result.rows).toEqual([])
      expect(result.totalCount).toBe(0)
    })

    it('should handle CSV with only headers', () => {
      const csv = 'product_name,category,price'

      const result = parseCSV(csv)

      expect(result.headers).toEqual(['product_name', 'category', 'price'])
      expect(result.rows).toEqual([])
      expect(result.totalCount).toBe(0)
    })

    it('should handle Thai characters', () => {
      const csv = `ชื่อสินค้า,หมวดหมู่
กล่องล็อค,เครื่องใช้ไฟฟ้า`

      const result = parseCSV(csv)

      expect(result.headers).toEqual(['ชื่อสินค้า', 'หมวดหมู่'])
      expect(result.rows[0]).toEqual({
        'ชื่อสินค้า': 'กล่องล็อค',
        'หมวดหมู่': 'เครื่องใช้ไฟฟ้า'
      })
    })
  })

  describe('validateCSV', () => {
    it('should validate valid CSV', () => {
      const parsed = {
        headers: ['product_name', 'category'],
        rows: [
          { product_name: 'Product 1', category: 'Cat 1' }
        ],
        totalCount: 1
      }

      const result = validateCSV(parsed)

      expect(result.isValid).toBe(true)
      expect(result.errors).toHaveLength(0)
    })

    it('should detect missing headers', () => {
      const parsed = {
        headers: [],
        rows: [],
        totalCount: 0
      }

      const result = validateCSV(parsed)

      expect(result.isValid).toBe(false)
      expect(result.errors).toContain('ไม่พบ headers ในไฟล์ CSV')
    })

    it('should warn about empty data', () => {
      const parsed = {
        headers: ['product_name'],
        rows: [],
        totalCount: 0
      }

      const result = validateCSV(parsed)

      expect(result.isValid).toBe(true)
      expect(result.warnings).toContain('ไม่พบข้อมูลในไฟล์ CSV')
    })

    it('should warn about duplicate headers', () => {
      const parsed = {
        headers: ['name', 'name', 'category'],
        rows: [
          { name: 'Product', category: 'Cat' }
        ],
        totalCount: 1
      }

      const result = validateCSV(parsed)

      expect(result.warnings).toContain('พบ headers ที่ซ้ำกัน')
    })

    it('should warn about empty headers', () => {
      const parsed = {
        headers: ['name', '', 'category'],
        rows: [],
        totalCount: 0
      }

      const result = validateCSV(parsed)

      expect(result.warnings.some(w => w.includes('headers ที่ว่าง'))).toBe(true)
    })
  })

  describe('getColumnStats', () => {
    it('should calculate column statistics', () => {
      const parsed = {
        headers: ['product_name', 'category'],
        rows: [
          { product_name: 'Product 1', category: 'Cat A' },
          { product_name: 'Product 2', category: 'Cat A' },
          { product_name: '', category: 'Cat B' },
          { product_name: 'Product 1', category: '' }
        ],
        totalCount: 4
      }

      const stats = getColumnStats(parsed, 'product_name')

      expect(stats.totalValues).toBe(4)
      expect(stats.emptyValues).toBe(1)
      expect(stats.uniqueValues).toBe(2) // Product 1, Product 2
      expect(stats.sampleValues).toHaveLength(2)
    })

    it('should handle empty column', () => {
      const parsed = {
        headers: ['name'],
        rows: [
          { name: '' },
          { name: '' }
        ],
        totalCount: 2
      }

      const stats = getColumnStats(parsed, 'name')

      expect(stats.totalValues).toBe(2)
      expect(stats.emptyValues).toBe(2)
      expect(stats.uniqueValues).toBe(0)
      expect(stats.sampleValues).toHaveLength(0)
    })

    it('should limit sample values to 5', () => {
      const parsed = {
        headers: ['name'],
        rows: Array.from({ length: 10 }, (_, i) => ({
          name: `Product ${i + 1}`
        })),
        totalCount: 10
      }

      const stats = getColumnStats(parsed, 'name')

      expect(stats.uniqueValues).toBe(10)
      expect(stats.sampleValues).toHaveLength(5)
    })
  })

  describe('detectEncoding', () => {
    it('should detect UTF-8 with Thai characters', () => {
      const text = 'สินค้าภาษาไทย'
      expect(detectEncoding(text)).toBe('UTF-8')
    })

    it('should detect UTF-8-BOM', () => {
      const text = '\uFEFFProduct Name'
      expect(detectEncoding(text)).toBe('UTF-8-BOM')
    })

    it('should return Unknown for non-Thai ASCII', () => {
      const text = 'Simple English Text'
      expect(detectEncoding(text)).toBe('Unknown')
    })
  })
})
