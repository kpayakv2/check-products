// Thai text processing utilities.
// clean() mirrors the normalization order of ThaiTextProcessor.process()
// in src/core/fresh_implementations.py (lowercase -> floating-vowel fix ->
// Thai digit conversion -> strip) so text cleaned here and text cleaned by
// the Python backend (which the FastAPI/Edge Function embedding path also
// normalizes) don't drift out of sync.
const THAI_DIGITS = '๐๑๒๓๔๕๖๗๘๙'
const ARABIC_DIGITS = '0123456789'
const THAI_TO_ARABIC = new Map(THAI_DIGITS.split('').map((d, i) => [d, ARABIC_DIGITS[i]]))

export class ThaiTextProcessor {
  static clean(text: string): string {
    let result = text.toLowerCase()

    // Fix floating vowel ordering (เ็ -> เ, แ็ -> แ)
    result = result.replace(/เ็/g, 'เ').replace(/แ็/g, 'แ')

    // Thai digits -> Arabic digits
    result = result.replace(/[๐-๙]/g, (d) => THAI_TO_ARABIC.get(d) ?? d)

    return result
      .replace(/[^\u0E00-\u0E7Fa-zA-Z0-9\s\-\.\(\)]/g, '')
      .replace(/\s+/g, ' ')
      .trim()
  }

  static tokenize(text: string): string[] {
    const tokens = text
      .split(/[\s\-\(\)\[\]\/\\,\.]+/)
      .filter(token => token.length >= 2)
    return [...new Set(tokens)]
  }

  static extractUnits(text: string): string[] {
    const unitPatterns = [
      /(\d+)\s*(กรัม|g|gram)/gi,
      /(\d+)\s*(มิลลิลิตร|ml|มล)/gi,
      /(\d+)\s*(ลิตร|l)/gi,
      /(\d+)\s*(กิโลกรม|kg|กก)/gi,
      /(\d+)\s*(ชิ้น|pcs)/gi,
      /(\d+)\s*(แพ็ค|pack)/gi,
      /(\d+)\s*(กล่อง|box)/gi
    ]

    const units: string[] = []
    unitPatterns.forEach(pattern => {
      const matches = text.match(pattern)
      if (matches) units.push(...matches)
    })
    return units
  }

  static extractAttributes(text: string): Record<string, any> {
    const attributes: Record<string, any> = {}

    const colors = ['แดง', 'เขียว', 'น้ำเงิน', 'เหลือง', 'ขาว', 'ดำ', 'ชมพู', 'ม่วง', 'ส้ม', 'เทา']
    const foundColors = colors.filter(color => text.includes(color))
    if (foundColors.length > 0) attributes.colors = foundColors

    const sizes = ['S', 'M', 'L', 'XL', 'XXL', 'เล็ก', 'กลาง', 'ใหญ่']
    const foundSizes = sizes.filter(size => text.includes(size))
    if (foundSizes.length > 0) attributes.sizes = foundSizes

    return attributes
  }
}
