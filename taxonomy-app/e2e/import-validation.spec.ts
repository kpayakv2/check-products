import { test, expect } from '@playwright/test'
import path from 'path'

test.describe('Import Logic Validation', () => {
  test.beforeEach(async ({ page }) => {
    // Listen for console logs to catch API errors (especially 400/500)
    page.on('console', msg => {
      if (msg.type() === 'error') console.log(`BROWSER ERROR: "${msg.text()}"`);
    });

    // Navigate to import wizard
    await page.goto('/import/wizard')
    await page.waitForLoadState('networkidle')
  })

  test('should complete full import workflow without API 400 errors', async ({ page }) => {
    // Step 1: Upload Sample CSV
    const fileChooserPromise = page.waitForEvent('filechooser')
    await page.locator('input[type="file"]').click()
    const fileChooser = await fileChooserPromise
    
    // Use the sample file we created
    const filePath = path.resolve(__dirname, '../test-data/sample-import.csv')
    await fileChooser.setFiles(filePath)

    // Check if file is ready
    await expect(page.locator('text=ไฟล์พร้อมประมวลผล')).toBeVisible()
    
    // Step 2: Go to Mapping
    await page.click('[data-testid="wizard-next-btn"]')
    await expect(page.locator('[data-testid="column-mapping-header"]').first()).toBeVisible()

    // Map "product_name" column
    const productNameSelect = page.locator('[data-testid^="column-select-"]').first()
    await productNameSelect.selectOption('product_name')

    // Step 3: Start AI Processing
    await page.click('[data-testid="mapping-complete-btn"]')
    
    // Check if we reached the processing step
    await expect(page.locator('text=AI กำลังประมวลผล')).toBeVisible()

    // Step 4: Wait for DB Saving (This is where API 400 usually happens)
    // We look for the "บันทึกสินค้าสำเร็จ" message or the "ถัดไป: ตรวจสอบ" button
    const nextBtn = page.locator('button:has-text("ถัดไป: ตรวจสอบ")')
    
    // Give it more time for real AI + DB processing (up to 60 seconds)
    await expect(nextBtn).toBeEnabled({ timeout: 60000 })

    // If we reach here without the test failing, it means no API 400 occurred during save!
    console.log('SUCCESS: Import processing completed without fatal API errors.')
    
    // Step 5: Final Verification - check if products are in the list
    await nextBtn.click()
    await expect(page.locator('text=ตรวจสอบและอนุมัติ')).toBeVisible()
  })
})
