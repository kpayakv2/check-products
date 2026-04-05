/**
 * CSV Parser Utilities
 * สำหรับอ่านและ parse ไฟล์ CSV จาก Product Similarity Checker
 */

export interface CSVRow {
  [key: string]: string
}

export interface ParsedCSV {
  headers: string[]
  rows: CSVRow[]
  totalCount: number
}

export interface CSVParseOptions {
  maxRows?: number
  skipEmptyLines?: boolean
  delimiter?: string
}

/**
 * Parse CSV text to structured data
 */
export function parseCSV(
  text: string,
  options: CSVParseOptions = {}
): ParsedCSV {
  const {
    maxRows,
    skipEmptyLines = true,
    delimiter = ','
  } = options

  const lines = text.split('\n').filter(line => 
    skipEmptyLines ? line.trim() !== '' : true
  )

  if (lines.length === 0) {
    return { headers: [], rows: [], totalCount: 0 }
  }

  // Parse headers
  const headers = lines[0]
    .split(delimiter)
    .map(h => h.trim().replace(/^"|"$/g, '')) // Remove quotes

  // Parse rows
  const dataLines = lines.slice(1)
  const rowsToParse = maxRows ? dataLines.slice(0, maxRows) : dataLines
  
  const rows: CSVRow[] = rowsToParse.map(line => {
    const values = parseCSVLine(line, delimiter)
    const row: CSVRow = {}
    
    headers.forEach((header, index) => {
      row[header] = values[index] || ''
    })
    
    return row
  })

  return {
    headers,
    rows,
    totalCount: dataLines.length
  }
}

/**
 * Parse single CSV line handling quoted values
 */
function parseCSVLine(line: string, delimiter: string = ','): string[] {
  const values: string[] = []
  let current = ''
  let inQuotes = false

  for (let i = 0; i < line.length; i++) {
    const char = line[i]
    const nextChar = line[i + 1]

    if (char === '"') {
      if (inQuotes && nextChar === '"') {
        // Escaped quote
        current += '"'
        i++
      } else {
        // Toggle quotes
        inQuotes = !inQuotes
      }
    } else if (char === delimiter && !inQuotes) {
      // End of value
      values.push(current.trim())
      current = ''
    } else {
      current += char
    }
  }

  // Add last value
  values.push(current.trim())

  return values
}

/**
 * Read file as text
 */
export async function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    
    reader.onload = (event) => {
      const text = event.target?.result as string
      resolve(text)
    }
    
    reader.onerror = () => {
      reject(new Error('Failed to read file'))
    }
    
    reader.readAsText(file, 'UTF-8')
  })
}

/**
 * Detect CSV encoding (simple check)
 */
export function detectEncoding(text: string): 'UTF-8' | 'UTF-8-BOM' | 'Unknown' {
  // Check for BOM
  if (text.charCodeAt(0) === 0xFEFF) {
    return 'UTF-8-BOM'
  }
  
  // Check for Thai characters
  if (/[\u0E00-\u0E7F]/.test(text)) {
    return 'UTF-8'
  }
  
  return 'Unknown'
}

/**
 * Validate CSV structure
 */
export function validateCSV(parsed: ParsedCSV): {
  isValid: boolean
  errors: string[]
  warnings: string[]
} {
  const errors: string[] = []
  const warnings: string[] = []

  // Check headers
  if (parsed.headers.length === 0) {
    errors.push('ไม่พบ headers ในไฟล์ CSV')
  }

  // Check for duplicate headers
  const headerSet = new Set(parsed.headers)
  if (headerSet.size !== parsed.headers.length) {
    warnings.push('พบ headers ที่ซ้ำกัน')
  }

  // Check rows
  if (parsed.rows.length === 0) {
    warnings.push('ไม่พบข้อมูลในไฟล์ CSV')
  }

  // Check for empty headers
  const emptyHeaders = parsed.headers.filter(h => !h || h.trim() === '')
  if (emptyHeaders.length > 0) {
    warnings.push(`พบ ${emptyHeaders.length} headers ที่ว่าง`)
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings
  }
}

/**
 * Get column statistics
 */
export function getColumnStats(parsed: ParsedCSV, columnName: string): {
  totalValues: number
  emptyValues: number
  uniqueValues: number
  sampleValues: string[]
} {
  const values = parsed.rows.map(row => row[columnName] || '')
  const nonEmptyValues = values.filter(v => v.trim() !== '')
  const uniqueSet = new Set(nonEmptyValues)

  return {
    totalValues: values.length,
    emptyValues: values.length - nonEmptyValues.length,
    uniqueValues: uniqueSet.size,
    sampleValues: Array.from(uniqueSet).slice(0, 5)
  }
}
