import { test, expect } from '@playwright/test'

test.describe('Synonym Management', () => {
  test.beforeEach(async ({ page }) => {
    // Listen for console logs to catch API errors
    page.on('console', msg => {
      if (msg.type() === 'error') console.log(`BROWSER ERROR: "${msg.text()}"`);
    });

    // Navigate to synonyms page
    await page.goto('/synonyms')
    
    // Wait for the page title to be visible
    await expect(page.locator('[data-testid="page-title"]')).toBeVisible()
  })

  test('should display synonym management interface', async ({ page }) => {
    // Check if the main heading is visible
    await expect(page.locator('[data-testid="page-title"]')).toContainText('Synonym Management')
    
    // Check if synonym list is visible
    await expect(page.locator('[data-testid="synonym-list"]')).toBeVisible()
  })

  test('should create new synonym', async ({ page }) => {
    const uniqueId = Math.floor(Math.random() * 1000000)
    const uniqueName = 'ชื่อพ้อง_ทดสอบ_' + uniqueId
    const uniqueCode = 'TEST_CODE_' + uniqueId
    
    // Click add synonym button
    await page.click('[data-testid="add-synonym-btn"]')
    
    // Fill in the form
    await page.fill('[data-testid="code-input"]', uniqueCode)
    await page.fill('[data-testid="lemma-input"]', uniqueName)
    
    // Add terms
    await page.click('[data-testid="add-term-btn"]')
    await page.fill('[data-testid="term-input-0"]', 'ตัวแปร_1_' + uniqueId)
    
    await page.click('[data-testid="add-term-btn"]')
    await page.fill('[data-testid="term-input-1"]', 'ตัวแปร_2_' + uniqueId)
    
    // Submit the form
    console.log(`Submitting new synonym: ${uniqueName} with code ${uniqueCode}`);
    await page.click('[data-testid="save-synonym-btn"]')
    
    // Wait for modal to close (Save button should be hidden)
    await expect(page.locator('[data-testid="save-synonym-btn"]')).toBeHidden({ timeout: 20000 })
    
    // Search for the newly created synonym
    await page.fill('[data-testid="search-input"]', uniqueName)
    await page.waitForTimeout(2000)
    
    // Verify it appears in the list
    await expect(page.locator(`text=${uniqueName}`)).toBeVisible()
  })

  test('should edit existing synonym', async ({ page }) => {
    // Wait for synonyms to load
    await page.waitForSelector('[data-testid^="synonym-item-"]', { timeout: 15000 })
    
    // Click edit button on first synonym
    const firstEditBtn = page.locator('[data-testid^="edit-synonym-"]').first()
    await firstEditBtn.click()
    
    // Update the name
    const uniqueUpdate = 'แก้ไข_' + Date.now()
    const lemmaInput = page.locator('[data-testid="lemma-input"]')
    await lemmaInput.clear()
    await lemmaInput.fill(uniqueUpdate)
    
    // Save changes
    await page.click('[data-testid="save-synonym-btn"]')
    
    // Wait for the modal to close
    await expect(page.locator('[data-testid="save-synonym-btn"]')).toBeHidden({ timeout: 20000 })
    
    // Verify the update in the list
    await page.fill('[data-testid="search-input"]', uniqueUpdate)
    await page.waitForTimeout(2000)
    await expect(page.locator(`text=${uniqueUpdate}`)).toBeVisible()
  })

  test('should delete synonym', async ({ page }) => {
    // Wait for synonyms to load
    await page.waitForSelector('[data-testid^="synonym-item-"]', { timeout: 15000 })
    
    // Setup dialog handler
    page.on('dialog', dialog => dialog.accept())
    
    // Get the name of the first synonym
    const firstItem = page.locator('[data-testid^="synonym-item-"]').first()
    const name = await firstItem.locator('[data-testid="synonym-lemma"]').textContent()
    
    // Click delete button
    const deleteBtn = page.locator('[data-testid^="delete-synonym-"]').first()
    await deleteBtn.click()
    
    // Wait for list to update (the item should disappear)
    if (name) {
      // Small timeout to allow DB processing
      await page.waitForTimeout(2000)
      await expect(page.locator(`text=${name}`)).not.toBeVisible({ timeout: 15000 })
    }
  })

  test('should search synonyms', async ({ page }) => {
    // Wait for synonyms to load
    await page.waitForSelector('[data-testid^="synonym-item-"]', { timeout: 15000 })
    
    // Enter search term
    const searchInput = page.locator('[data-testid="search-input"]')
    await searchInput.fill('ชื่อพ้อง')
    
    // Wait for UI to filter
    await page.waitForTimeout(2000)
    
    // Check results count change
    const visibleCount = await page.locator('[data-testid^="synonym-item-"]:visible').count()
    console.log(`Found ${visibleCount} visible synonyms for "ชื่อพ้อง"`)
    
    // Clear search and type nonsense
    await searchInput.clear()
    await searchInput.fill('something_that_definitely_does_not_exist_' + Date.now())
    await page.waitForTimeout(2000)
    
    const countAfter = await page.locator('[data-testid^="synonym-item-"]:visible').count()
    expect(countAfter).toBe(0)
  })

  test('should handle loading states', async ({ page }) => {
    await page.reload()
    await page.waitForLoadState('networkidle')
    const loading = page.locator('[data-testid="loading-indicator"]')
    if (await loading.isVisible()) {
      await expect(loading).toBeHidden({ timeout: 20000 })
    }
    await expect(page.locator('[data-testid="page-title"]')).toBeVisible()
  })
})
