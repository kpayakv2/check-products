import { test, expect } from '@playwright/test'
import { existsSync, readFileSync } from 'fs'
import path from 'path'

/** อ่าน INTERNAL_API_SECRET จาก .env.local — playwright.config ไม่ได้โหลด .env ให้ */
function readInternalSecret(): string | null {
  if (process.env.INTERNAL_API_SECRET) return process.env.INTERNAL_API_SECRET

  const envPath = path.join(__dirname, '..', '.env.local')
  if (!existsSync(envPath)) return null

  const match = readFileSync(envPath, 'utf8').match(/^INTERNAL_API_SECRET=(.+)$/m)
  return match ? match[1].trim() : null
}

/**
 * หน้าตรวจซ้ำหมวดหมู่สินค้าเก่า (Data Quality → Recheck)
 *
 * ต้องมีข้อมูลจากสองสคริปต์นี้ก่อน ไม่งั้นจะไม่มีรายการให้ตรวจ:
 *   .venv/Scripts/python.exe scripts/import_legacy_products.py
 *   .venv/Scripts/python.exe scripts/recheck_legacy_categories.py
 *
 * การกดยืนยันต้องมี session cookie (middleware.ts กัน POST ไว้)
 * ตั้ง INTERNAL_API_SECRET ใน taxonomy-app/.env.local แล้วปลดล็อกที่ /unlock ก่อน
 */
test.describe('Recheck legacy categories', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/data-quality')
    await page.waitForLoadState('networkidle')
    await page.locator('[data-testid="tab-recheck"]').click()

    // ต้องรอให้โหลดรายการเสร็จก่อน ไม่งั้นนับแถวได้ 0 ทั้งที่มีข้อมูล
    await expect(page.locator('[data-testid="recheck-loading"]')).toHaveCount(0)
    await expect(
      page.locator('[data-testid^="recheck-row-"]').first().or(
        page.locator('[data-testid="recheck-empty"]')
      )
    ).toBeVisible()
  })

  test('แสดงหมวดที่คนจัดคู่กับหมวดที่ AI เสนอ', async ({ page }) => {
    await expect(page.locator('[data-testid="recheck-tab"]')).toBeVisible()
    await expect(page.locator('[data-testid="recheck-total"]')).toContainText('เหลือ')

    const rows = page.locator('[data-testid^="recheck-row-"]')
    const count = await rows.count()
    if (count === 0) {
      // ตรวจครบแล้วก็ถือว่าหน้าทำงานถูกต้อง
      await expect(page.locator('[data-testid="recheck-empty"]')).toBeVisible()
      return
    }

    // แถวแรกต้องมีปุ่มครบทั้งสามทาง
    const first = rows.first()
    await expect(first.locator('[data-testid^="recheck-keep-"]')).toBeVisible()
    await expect(first.locator('[data-testid^="recheck-accept-"]')).toBeVisible()
    await expect(first.locator('[data-testid^="recheck-override-"]')).toBeVisible()
  })

  test('กดตัดสินแล้วรายการหายไปจากคิว', async ({ page, context }) => {
    // การบันทึกเป็น POST ซึ่ง middleware.ts บังคับให้มี session — ต้องปลดล็อกก่อน
    const secret = readInternalSecret()
    test.skip(!secret, 'ต้องตั้ง INTERNAL_API_SECRET ใน taxonomy-app/.env.local ก่อน')

    const unlock = await context.request.post('/api/unlock', { data: { secret } })
    expect(unlock.ok(), 'ปลดล็อกไม่สำเร็จ — ตรวจ INTERNAL_API_SECRET').toBe(true)
    await page.reload()
    await page.locator('[data-testid="tab-recheck"]').click()
    await expect(page.locator('[data-testid="recheck-loading"]')).toHaveCount(0)

    const rows = page.locator('[data-testid^="recheck-row-"]')
    if ((await rows.count()) === 0) test.skip()

    const before = await rows.count()
    const firstId = await rows.first().getAttribute('data-testid')

    await rows.first().locator('[data-testid^="recheck-keep-"]').click()

    // แถวที่ตัดสินแล้วต้องหายทันที ไม่ต้องรีเฟรชทั้งหน้า
    await expect(page.locator(`[data-testid="${firstId}"]`)).toHaveCount(0)
    await expect(rows).toHaveCount(before - 1)
  })

  test('ไม่มี console error และไม่เลื่อนแนวนอนที่จอ 375px', async ({ page }) => {
    const errors: string[] = []
    page.on('console', (message) => {
      // favicon 404 มีอยู่เดิมทั้งเว็บ ไม่เกี่ยวกับหน้านี้
      if (message.type() === 'error' && !message.text().includes('favicon')) {
        errors.push(message.text())
      }
    })

    await page.setViewportSize({ width: 375, height: 800 })
    await page.waitForLoadState('networkidle')

    const scrollsHorizontally = await page.evaluate(
      () => document.documentElement.scrollWidth > document.documentElement.clientWidth
    )
    expect(scrollsHorizontally).toBe(false)
    expect(errors).toEqual([])
  })
})
