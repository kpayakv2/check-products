import { test, expect } from '@playwright/test'
import { setupTestDatabase, clearTestData } from '../__tests__/setup/database-setup'

test.describe('Real User Workflows E2E Tests', () => {
  test.beforeAll(async () => {
    await setupTestDatabase()
  })

  test.afterAll(async () => {
    await clearTestData()
  })

  test.beforeEach(async ({ page }) => {
    await clearTestData()
    await setupTestDatabase()
    await page.goto('http://localhost:3000')
  })

  test.describe('Taxonomy Management Workflow', () => {
    test('should complete full taxonomy management workflow', async ({ page }) => {
      // Navigate to taxonomy page
      await page.click('text=Taxonomy')
      await expect(page).toHaveURL(/.*taxonomy/)

      // Should see existing test data
      await expect(page.locator('[data-testid="taxonomy-tree"]')).toBeVisible()
      await expect(page.locator('text=Test Electronics')).toBeVisible()
      await expect(page.locator('text=Test Smartphones')).toBeVisible()

      // Add new root category
      await page.click('[data-testid="add-category"]')
      await page.fill('[data-testid="category-name"]', 'E2E Test Category')
      await page.fill('[data-testid="category-code"]', 'E2E001')
      await page.click('[data-testid="save-category"]')

      // Verify new category appears
      await expect(page.locator('text=E2E Test Category')).toBeVisible()

      // Add child category
      await page.hover('text=E2E Test Category')
      await page.click('[data-testid="add-child-E2E001"]')
      await page.fill('[data-testid="category-name"]', 'E2E Child Category')
      await page.fill('[data-testid="category-code"]', 'E2E002')
      await page.click('[data-testid="save-category"]')

      // Verify hierarchy
      await expect(page.locator('text=E2E Child Category')).toBeVisible()
      
      // Check indentation/level
      const childElement = page.locator('[data-testid="node-E2E002"]')
      await expect(childElement).toHaveAttribute('data-level', '2')

      // Edit category
      await page.hover('text=E2E Test Category')
      await page.click('[data-testid="edit-E2E001"]')
      await page.fill('[data-testid="category-name"]', 'E2E Updated Category')
      await page.click('[data-testid="save-category"]')

      // Verify update
      await expect(page.locator('text=E2E Updated Category')).toBeVisible()
      await expect(page.locator('text=E2E Test Category')).not.toBeVisible()

      // Delete child first (foreign key constraint)
      await page.hover('text=E2E Child Category')
      await page.click('[data-testid="delete-E2E002"]')
      await page.click('[data-testid="confirm-delete"]')

      // Verify child deleted
      await expect(page.locator('text=E2E Child Category')).not.toBeVisible()

      // Delete parent
      await page.hover('text=E2E Updated Category')
      await page.click('[data-testid="delete-E2E001"]')
      await page.click('[data-testid="confirm-delete"]')

      // Verify parent deleted
      await expect(page.locator('text=E2E Updated Category')).not.toBeVisible()
    })

    test('should handle taxonomy search and filtering', async ({ page }) => {
      await page.goto('http://localhost:3000/taxonomy')

      // Search for existing category
      await page.fill('[data-testid="search-input"]', 'Electronics')
      await page.press('[data-testid="search-input"]', 'Enter')

      // Should show only matching results
      await expect(page.locator('text=Test Electronics')).toBeVisible()
      await expect(page.locator('text=Test Smartphones')).not.toBeVisible()

      // Clear search
      await page.fill('[data-testid="search-input"]', '')
      await page.press('[data-testid="search-input"]', 'Enter')

      // Should show all categories
      await expect(page.locator('text=Test Electronics')).toBeVisible()
      await expect(page.locator('text=Test Smartphones')).toBeVisible()

      // Filter by level
      await page.selectOption('[data-testid="level-filter"]', '2')

      // Should show only level 2 categories
      await expect(page.locator('text=Test Smartphones')).toBeVisible()
      await expect(page.locator('text=Test Electronics')).not.toBeVisible()
    })
  })

  test.describe('Synonym Management Workflow', () => {
    test('should complete full synonym management workflow', async ({ page }) => {
      await page.goto('http://localhost:3000/synonyms')

      // Should see existing test synonyms
      await expect(page.locator('text=test-smartphone')).toBeVisible()

      // Add new synonym
      await page.click('[data-testid="add-synonym"]')
      await page.fill('[data-testid="lemma-input"]', 'e2e-laptop')
      await page.selectOption('[data-testid="category-select"]', 'test-tax-2')
      await page.fill('[data-testid="confidence-input"]', '0.88')

      // Add terms
      await page.click('[data-testid="add-term"]')
      await page.fill('[data-testid="term-input-0"]', 'notebook computer')
      await page.selectOption('[data-testid="language-select-0"]', 'en')

      await page.click('[data-testid="add-term"]')
      await page.fill('[data-testid="term-input-1"]', 'แล็ปท็อป')
      await page.selectOption('[data-testid="language-select-1"]', 'th')

      await page.click('[data-testid="save-synonym"]')

      // Verify new synonym appears
      await expect(page.locator('text=e2e-laptop')).toBeVisible()
      await expect(page.locator('text=notebook computer')).toBeVisible()
      await expect(page.locator('text=แล็ปท็อป')).toBeVisible()

      // Edit synonym
      await page.hover('text=e2e-laptop')
      await page.click('[data-testid="edit-synonym"]')
      await page.fill('[data-testid="confidence-input"]', '0.95')
      await page.check('[data-testid="verified-checkbox"]')
      await page.click('[data-testid="save-synonym"]')

      // Verify update
      const synonymRow = page.locator('[data-testid="synonym-e2e-laptop"]')
      await expect(synonymRow.locator('text=0.95')).toBeVisible()
      await expect(synonymRow.locator('[data-testid="verified-badge"]')).toBeVisible()

      // Delete synonym
      await page.hover('text=e2e-laptop')
      await page.click('[data-testid="delete-synonym"]')
      await page.click('[data-testid="confirm-delete"]')

      // Verify deletion
      await expect(page.locator('text=e2e-laptop')).not.toBeVisible()
    })

    test('should handle CSV import/export', async ({ page }) => {
      await page.goto('http://localhost:3000/synonyms')

      // Test CSV export
      await page.click('[data-testid="export-csv"]')
      
      // Wait for download
      const downloadPromise = page.waitForEvent('download')
      await page.click('[data-testid="confirm-export"]')
      const download = await downloadPromise

      // Verify download
      expect(download.suggestedFilename()).toContain('synonyms')
      expect(download.suggestedFilename()).toContain('.csv')

      // Test CSV import
      await page.click('[data-testid="import-csv"]')
      
      // Upload test CSV file
      const fileInput = page.locator('[data-testid="csv-file-input"]')
      await fileInput.setInputFiles('__tests__/fixtures/test-synonyms.csv')
      
      await page.click('[data-testid="upload-csv"]')

      // Verify import success
      await expect(page.locator('[data-testid="import-success"]')).toBeVisible()
      await expect(page.locator('text=Import completed successfully')).toBeVisible()
    })
  })

  test.describe('Product Review Workflow', () => {
    test('should complete product review workflow', async ({ page }) => {
      await page.goto('http://localhost:3000/products')

      // Should see test products
      await expect(page.locator('text=Test iPhone 15 Pro')).toBeVisible()

      // Select product for review
      await page.click('[data-testid="product-test-prod-1"]')

      // Should open detail panel
      await expect(page.locator('[data-testid="product-detail-panel"]')).toBeVisible()
      await expect(page.locator('text=Test iPhone 15 Pro')).toBeVisible()
      await expect(page.locator('text=Test latest iPhone model')).toBeVisible()

      // Check similarity score
      await expect(page.locator('text=85%')).toBeVisible()

      // Approve product using button
      await page.click('[data-testid="approve-button"]')

      // Verify status change
      await expect(page.locator('[data-testid="status-approved"]')).toBeVisible()
      await expect(page.locator('[data-testid="success-message"]')).toBeVisible()

      // Test keyboard shortcuts
      await page.keyboard.press('r') // Reject shortcut
      await expect(page.locator('[data-testid="status-rejected"]')).toBeVisible()

      await page.keyboard.press('a') // Approve shortcut
      await expect(page.locator('[data-testid="status-approved"]')).toBeVisible()

      // Navigate with arrow keys
      await page.keyboard.press('ArrowDown')
      // Should select next product (if available)

      await page.keyboard.press('Escape')
      // Should close detail panel
      await expect(page.locator('[data-testid="product-detail-panel"]')).not.toBeVisible()
    })

    test('should handle batch operations', async ({ page }) => {
      // Add more test products first
      await setupTestDatabase()
      
      await page.goto('http://localhost:3000/products')

      // Select multiple products
      await page.check('[data-testid="select-test-prod-1"]')
      
      // Batch approve
      await page.click('[data-testid="batch-approve"]')
      await page.click('[data-testid="confirm-batch-action"]')

      // Verify batch update
      await expect(page.locator('[data-testid="batch-success"]')).toBeVisible()
      await expect(page.locator('text=1 products approved')).toBeVisible()
    })
  })

  test.describe('Import System Workflow', () => {
    test('should complete product import workflow', async ({ page }) => {
      await page.goto('http://localhost:3000/import')

      // Upload test file
      const fileInput = page.locator('[data-testid="file-input"]')
      await fileInput.setInputFiles('__tests__/fixtures/test-products.xlsx')

      // Start import
      await page.click('[data-testid="start-import"]')

      // Should show progress
      await expect(page.locator('[data-testid="import-progress"]')).toBeVisible()

      // Wait for completion (with timeout)
      await expect(page.locator('[data-testid="import-complete"]')).toBeVisible({ timeout: 30000 })

      // Verify results
      await expect(page.locator('[data-testid="import-summary"]')).toBeVisible()
      await expect(page.locator('text=Import completed successfully')).toBeVisible()

      // Check imported products
      await page.click('[data-testid="view-imported-products"]')
      await expect(page).toHaveURL(/.*products/)
      
      // Should see newly imported products
      await expect(page.locator('[data-testid="imported-products"]')).toBeVisible()
    })
  })

  test.describe('Cross-Feature Integration', () => {
    test('should handle complete taxonomy-to-product workflow', async ({ page }) => {
      // 1. Create taxonomy category
      await page.goto('http://localhost:3000/taxonomy')
      await page.click('[data-testid="add-category"]')
      await page.fill('[data-testid="category-name"]', 'E2E Integration Test')
      await page.fill('[data-testid="category-code"]', 'INT001')
      await page.click('[data-testid="save-category"]')

      // 2. Create synonym for the category
      await page.goto('http://localhost:3000/synonyms')
      await page.click('[data-testid="add-synonym"]')
      await page.fill('[data-testid="lemma-input"]', 'integration-test')
      await page.selectOption('[data-testid="category-select"]', 'INT001')
      await page.click('[data-testid="save-synonym"]')

      // 3. Import products that should match
      await page.goto('http://localhost:3000/import')
      await fileInput.setInputFiles('__tests__/fixtures/integration-test-products.xlsx')
      await page.click('[data-testid="start-import"]')
      await expect(page.locator('[data-testid="import-complete"]')).toBeVisible({ timeout: 30000 })

      // 4. Review matched products
      await page.goto('http://localhost:3000/products')
      await expect(page.locator('[data-testid="category-INT001"]')).toBeVisible()

      // 5. Verify AI suggestions worked
      await page.click('[data-testid="product-with-category-INT001"]')
      await expect(page.locator('[data-testid="ai-suggestion"]')).toBeVisible()
      await expect(page.locator('text=integration-test')).toBeVisible()
    })
  })
})
