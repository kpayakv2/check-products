import { test, expect, type Page, type Locator } from '@playwright/test'

/**
 * หน้า /synonyms ถูกยุบเข้าไปเป็นแท็บหนึ่งของ /taxonomy แล้ว
 * spec เดิมยังยิงไปที่ /synonyms จึงตกตั้งแต่ beforeEach ทุกเทสต์
 *
 * ชุดนี้ยิงใส่ฐานข้อมูลจริง เทสต์ที่แก้หรือลบจึงต้องสร้างข้อมูลของตัวเองขึ้นมาก่อน
 * แล้วเก็บกวาดท้ายเทสต์ ห้ามไปแก้/ลบ synonym แถวแรกที่บังเอิญเจอ
 */

const unique = () => `${Date.now()}_${Math.floor(Math.random() * 100000)}`

/**
 * กดแท็บแบบทนการ hydrate ช้า — ปุ่มถูกวาดออกมาก่อนที่ React จะผูก onClick
 * คลิกที่ลงไปก่อนหน้านั้นจะเงียบหายไปเฉย ๆ ไม่มี error ให้จับ
 * จึงต้องกดซ้ำจนแท็บเปลี่ยนจริง แทนที่จะกดครั้งเดียวแล้วรอ
 */
async function openSynonymsTab(page: Page) {
  await page.goto('/taxonomy')
  await expect(async () => {
    await page.click('[data-testid="tab-synonyms"]')
    await expect(page.locator('[data-testid="page-title"]')).toBeVisible({ timeout: 3000 })
  }).toPass({ timeout: 30000 })
}

/** เหมือน openSynonymsTab แต่ไม่ goto ซ้ำ ใช้หลัง page.reload() */
async function openSynonymsTabAfterReload(page: Page) {
  await expect(async () => {
    await page.click('[data-testid="tab-synonyms"]')
    await expect(page.locator('[data-testid="page-title"]')).toBeVisible({ timeout: 3000 })
  }).toPass({ timeout: 30000 })
}

function itemFor(page: Page, lemma: string): Locator {
  return page.locator('[data-testid^="synonym-item-"]').filter({ hasText: lemma })
}

/** ค้นหาด้วยชื่อที่ไม่ซ้ำ เพื่อให้ยืนยันผลได้โดยไม่ขึ้นกับจำนวนแถวทั้งหมด */
async function searchFor(page: Page, term: string) {
  const search = page.locator('[data-testid="search-input"]')
  await search.clear()
  await search.fill(term)
}

/**
 * POST /api/synonyms ตอบ 400 ถ้า `terms` ว่าง (ต้องมีคำพ้องอย่างน้อย 1 คำ)
 * ฟอร์มฝั่งหน้าเว็บไม่ได้บังคับข้อนี้ กดบันทึกแล้วได้แค่ toast แดงลอย ๆ
 * ทุกเทสต์จึงต้องใส่ term มาอย่างน้อยหนึ่งคำเสมอ
 */
async function createSynonym(page: Page, lemma: string, code: string, terms: string[]) {
  await page.click('[data-testid="add-synonym-btn"]')
  await page.fill('[data-testid="code-input"]', code)
  await page.fill('[data-testid="lemma-input"]', lemma)

  for (let i = 0; i < terms.length; i++) {
    await page.click('[data-testid="add-term-btn"]')
    await page.fill(`[data-testid="term-input-${i}"]`, terms[i])
  }

  await page.click('[data-testid="save-synonym-btn"]')
  await expect(page.locator('[data-testid="save-synonym-btn"]')).toBeHidden({ timeout: 20000 })
}

/**
 * เก็บกวาดใน finally — ห้าม throw ทับ error ตัวจริงของเทสต์
 * ถ้าเก็บกวาดไม่สำเร็จให้บอกไว้ใน log แล้วปล่อยผ่าน จะได้เห็นสาเหตุที่เทสต์ตกจริง ๆ
 */
async function cleanUp(page: Page, lemma: string) {
  try {
    await removeSynonym(page, lemma)
  } catch (error) {
    console.warn(`เก็บกวาด "${lemma}" ไม่สำเร็จ ต้องลบด้วยมือ:`, error)
  }
}

/** เรียกได้แม้แถวถูกลบไปแล้ว */
async function removeSynonym(page: Page, lemma: string) {
  await searchFor(page, lemma)
  const item = itemFor(page, lemma)
  if ((await item.count()) === 0) return

  page.once('dialog', dialog => dialog.accept())
  await item.first().locator('[data-testid^="delete-synonym-"]').click()
  await expect(itemFor(page, lemma)).toHaveCount(0, { timeout: 15000 })
}

test.describe('Synonym Management', () => {
  test.beforeEach(async ({ page }) => {
    page.on('console', msg => {
      if (msg.type() === 'error') console.log(`BROWSER ERROR: "${msg.text()}"`)
    })
    await openSynonymsTab(page)
  })

  test('should display synonym management interface', async ({ page }) => {
    await expect(page.locator('[data-testid="page-title"]')).toContainText('Synonym Management')
    await expect(page.locator('[data-testid="synonym-list"]')).toBeVisible()
  })

  test('should create new synonym', async ({ page }) => {
    const id = unique()
    const lemma = `ชื่อพ้อง_ทดสอบ_${id}`

    try {
      await createSynonym(page, lemma, `TEST_CODE_${id}`, [
        `ตัวแปร_1_${id}`,
        `ตัวแปร_2_${id}`,
      ])

      await searchFor(page, lemma)
      await expect(itemFor(page, lemma)).toHaveCount(1)
    } finally {
      await cleanUp(page, lemma)
    }
  })

  test('should edit existing synonym', async ({ page }) => {
    const id = unique()
    const lemma = `ชื่อพ้อง_แก้ไข_${id}`
    const renamed = `แก้ไขแล้ว_${id}`

    try {
      await createSynonym(page, lemma, `EDIT_CODE_${id}`, [`คำพ้อง_${id}`])

      await searchFor(page, lemma)
      await itemFor(page, lemma).first().locator('[data-testid^="edit-synonym-"]').click()

      const lemmaInput = page.locator('[data-testid="lemma-input"]')
      await lemmaInput.clear()
      await lemmaInput.fill(renamed)
      await page.click('[data-testid="save-synonym-btn"]')
      await expect(page.locator('[data-testid="save-synonym-btn"]')).toBeHidden({ timeout: 20000 })

      await searchFor(page, renamed)
      await expect(itemFor(page, renamed)).toHaveCount(1)

      await searchFor(page, lemma)
      await expect(itemFor(page, lemma)).toHaveCount(0)
    } finally {
      await cleanUp(page, lemma)
      await cleanUp(page, renamed)
    }
  })

  test('should delete synonym', async ({ page }) => {
    const id = unique()
    const lemma = `ชื่อพ้อง_จะลบ_${id}`

    await createSynonym(page, lemma, `DEL_CODE_${id}`, [`คำพ้อง_${id}`])
    await searchFor(page, lemma)
    await expect(itemFor(page, lemma)).toHaveCount(1)

    page.once('dialog', dialog => dialog.accept())
    await itemFor(page, lemma).first().locator('[data-testid^="delete-synonym-"]').click()

    await expect(itemFor(page, lemma)).toHaveCount(0, { timeout: 15000 })

    // ยืนยันว่าหายจริงจากฐานข้อมูล ไม่ใช่แค่หายจาก state ในหน้า
    await page.reload()
    await openSynonymsTabAfterReload(page)
    await searchFor(page, lemma)
    await expect(itemFor(page, lemma)).toHaveCount(0)
  })

  test('should search synonyms', async ({ page }) => {
    const id = unique()
    const lemma = `ชื่อพ้อง_ค้นหา_${id}`

    try {
      await createSynonym(page, lemma, `FIND_CODE_${id}`, [`คำพ้อง_${id}`])

      await searchFor(page, lemma)
      await expect(itemFor(page, lemma)).toHaveCount(1)

      await searchFor(page, `ไม่มีคำนี้แน่นอน_${id}`)
      await expect(page.locator('[data-testid^="synonym-item-"]')).toHaveCount(0)
    } finally {
      await cleanUp(page, lemma)
    }
  })

  test('should handle loading states', async ({ page }) => {
    await page.reload()
    // reload แล้วแท็บกลับไปเป็น tree เสมอ ต้องกดเข้าแท็บ synonyms ใหม่
    await openSynonymsTabAfterReload(page)

    const loading = page.locator('[data-testid="loading-indicator"]')
    if (await loading.isVisible()) {
      await expect(loading).toBeHidden({ timeout: 20000 })
    }
    await expect(page.locator('[data-testid="page-title"]')).toBeVisible()
  })
})
