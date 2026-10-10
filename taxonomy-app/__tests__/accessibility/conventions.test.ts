/**
 * @jest-environment node
 *
 * กฎรวมของงาน WCAG 2.2 AA ที่ตรวจจากซอร์สได้โดยไม่ต้องเปิดเบราว์เซอร์
 * (docs/superpowers/plans/2026-09-19-wcag-fix-split.md ขั้นที่ 0 ข้อ 2)
 *
 * ตรวจจากซอร์สเพราะสถานะที่เกิดชั่วคราว เช่น หน้ากำลังโหลด หรือข้อความที่โผล่ตอน hover
 * เปิดเบราว์เซอร์ไปก็ไม่เจอในสภาพปกติ แต่ซอร์สเห็นทุกจุดเสมอ
 */
import { readdirSync, readFileSync, statSync } from 'fs'
import path from 'path'

const ROOT = path.resolve(__dirname, '..', '..')

// หน้านี้อยู่นอกขอบเขตงาน WCAG รอบนี้ (ไม่ได้อยู่ในรายงานตรวจ) — ถ้าจะเก็บให้ครบ ลบบรรทัดนี้ออก
const OUT_OF_SCOPE = [path.join('app', 'ai-brain')]

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const full = path.join(dir, name)
    if (statSync(full).isDirectory()) return sourceFiles(full)
    return full.endsWith('.tsx') ? [full] : []
  })
}

const files = ['app', 'components']
  .flatMap((dir) => sourceFiles(path.join(ROOT, dir)))
  .filter((file) => !OUT_OF_SCOPE.some((skip) => file.includes(skip)))

/** คืน "ไฟล์:บรรทัด: ข้อความ" ของทุกบรรทัดที่เข้าเงื่อนไข */
function findLines(test: (line: string) => boolean): string[] {
  return files.flatMap((file) =>
    readFileSync(file, 'utf8')
      .split(/\r?\n/)
      .flatMap((line, index) =>
        test(line)
          ? [`${path.relative(ROOT, file)}:${index + 1}: ${line.trim().slice(0, 110)}`]
          : []
      )
  )
}

describe('WCAG 2.2 AA conventions', () => {
  it('has no text below the 12px floor', () => {
    const offenders = findLines((line) =>
      [...line.matchAll(/text-\[(\d+(?:\.\d+)?)px\]/g)].some((m) => Number(m[1]) < 12)
    )
    expect(offenders).toEqual([])
  })

  it('only loops pulse and ping animations for users who have not asked for reduced motion', () => {
    // animate-spin ไม่รวม: เป็นตัวบอกว่ากำลังโหลด ซึ่ง WCAG ถือเป็นการเคลื่อนไหวที่จำเป็น
    // ต้องนำหน้าด้วย motion-safe: เท่านั้น — ตัวแปรอื่นอย่าง group-hover: หรือ md: ไม่ช่วยอะไร
    const offenders = findLines(
      (line) => /(?<!motion-safe:)\banimate-(pulse|ping)\b/.test(line)
    )
    expect(offenders).toEqual([])
  })

  it('never hides text or controls behind hover alone', () => {
    // opacity-0 + group-hover:opacity-100 แปลว่าคีย์บอร์ดกับหน้าจอสัมผัสไม่มีวันเห็น
    // ตรวจเฉพาะข้อความและปุ่ม — ลูกศรหรือพื้นไล่สีที่แค่ประดับตอน hover ไม่ใช่ข้อมูล
    // ถ้าซ่อนไว้จริงๆ ต้องมี group-focus-within หรือ focus-visible ให้ทางอื่นเปิดได้
    const offenders = findLines(
      (line) =>
        /<(p|span|h[1-6]|button|a|label|li|td)\b/.test(line) &&
        /\bopacity-0\b/.test(line) &&
        /group-hover:opacity-100/.test(line) &&
        !/group-focus-within:opacity-100|focus-visible:opacity-100|focus:opacity-100/.test(line)
    )
    expect(offenders).toEqual([])
  })
})
