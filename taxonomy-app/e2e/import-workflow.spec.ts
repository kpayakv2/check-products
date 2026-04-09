import { test, expect } from '@playwright/test';

test.describe('Import Workflow', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to import landing page
    await page.goto('/import');
    
    // Wait for page to load
    await page.waitForLoadState('domcontentloaded');
  });

  test('should display import landing page', async ({ page }) => {
    // Check if the main heading is visible (using a more specific selector to avoid ambiguity)
    await expect(page.locator('[data-testid="import-title"]')).toBeVisible();
    
    // Check if the main action cards are visible
    await expect(page.locator('[data-testid="new-import-card"]')).toBeVisible();
    await expect(page.locator('[data-testid="pending-reviews-card"]')).toBeVisible();
  });

  test('should navigate to wizard and handle file upload', async ({ page }) => {
    // Go to wizard
    await page.goto('/import/wizard');
    await page.waitForLoadState('domcontentloaded');

    // Check if the wizard is displayed
    await expect(page.locator('text=เลือกวิธีการ Import')).toBeVisible();
    
    // Create a mock CSV content
    const csvContent = `product_name,category,price\nProduct A,Electronics,1000\nProduct B,Clothing,500`;

    // Set file input
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles({
      name: 'test-import.csv',
      mimeType: 'text/csv',
      buffer: Buffer.from(csvContent)
    });

    // Check if file is recognized
    await expect(page.locator('text=test-import.csv')).toBeVisible();
    await expect(page.locator('text=ไฟล์พร้อมประมวลผล')).toBeVisible();

    // Click next to go to mapping
    await page.click('[data-testid="wizard-next-btn"]');

    // Wait for mapping step
    await expect(page.locator('[data-testid="column-mapping-header"]').first()).toBeVisible();
    
    // Select column mapping for product_name
    const productNameSelect = page.locator('[data-testid^="column-select-"]').first();
    if (await productNameSelect.isVisible()) {
      await productNameSelect.selectOption('product_name');
    }

    // Click next to go to processing
    await page.click('[data-testid="mapping-complete-btn"]');
  });
});
