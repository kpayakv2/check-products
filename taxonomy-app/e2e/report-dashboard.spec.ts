import { test, expect } from '@playwright/test';

test.describe('Reports and Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to reports page
    await page.goto('/reports');
    
    // Wait for page to load (it has an 800ms loading state)
    await page.waitForSelector('[data-testid="reports-title"]', { timeout: 5000 });
  });

  test('should display main dashboard elements', async ({ page }) => {
    // Check if the main heading is visible
    await expect(page.locator('[data-testid="reports-title"]')).toBeVisible();
    
    // Check for key metric cards
    await expect(page.locator('[data-testid="accuracy-card"]')).toBeVisible();
    await expect(page.locator('[data-testid="velocity-card"]')).toBeVisible();
  });

  test('should display charts and data visualization', async ({ page }) => {
    // Check for the mock bar chart elements
    await expect(page.locator('[data-testid="heatmap-chart"]')).toBeVisible();
    const bars = page.locator('[data-testid^="chart-bar-"]');
    const count = await bars.count();
    expect(count).toBeGreaterThan(0);
  });

  test('should show activity log', async ({ page }) => {
    await expect(page.locator('[data-testid="activity-log"]')).toBeVisible();
    await expect(page.locator('text=/Auto-Classification/i')).toBeVisible();
  });
});
