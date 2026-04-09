import { test, expect } from '@playwright/test'

test.describe('Product Review', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to products page
    await page.goto('/products')
    
    // Wait for page to load
    await page.waitForLoadState('networkidle')
  })

  test('should display product review interface', async ({ page }) => {
    // Check if the main heading is visible
    await expect(page.locator('h1')).toContainText('Product Review')
    
    // Check if product table is visible
    await expect(page.locator('[data-testid="product-table"]')).toBeVisible()
    
    // Wait for products to load
    await page.waitForTimeout(2000)
    
    // Check if at least one product row is visible
    const productRows = page.locator('[data-testid^="product-row-"]')
    await expect(productRows.first()).toBeVisible()
  })

  test('should approve product', async ({ page }) => {
    // Wait for products to load
    await page.waitForSelector('[data-testid^="product-row-"]')
    
    // Click on first product to select it
    const firstProduct = page.locator('[data-testid^="product-row-"]').first()
    await firstProduct.click()
    
    // Click approve button
    await page.click('[data-testid="approve-btn"]')
    
    // Wait for status update
    await page.waitForTimeout(1000)
    
    // Check if status changed to approved
    await expect(page.locator('[data-testid="product-status"]')).toContainText('approved')
    
    // Should show success notification
    await expect(page.locator('.toast')).toContainText('approved')
  })

  test('should reject product', async ({ page }) => {
    // Wait for products to load
    await page.waitForSelector('[data-testid^="product-row-"]')
    
    // Click on first product to select it
    const firstProduct = page.locator('[data-testid^="product-row-"]').first()
    await firstProduct.click()
    
    // Click reject button
    await page.click('[data-testid="reject-btn"]')
    
    // Wait for status update
    await page.waitForTimeout(1000)
    
    // Check if status changed to rejected
    await expect(page.locator('[data-testid="product-status"]')).toContainText('rejected')
    
    // Should show success notification
    await expect(page.locator('.toast')).toContainText('rejected')
  })

  test('should use keyboard shortcuts', async ({ page }) => {
    // Wait for products to load
    await page.waitForSelector('[data-testid^="product-row-"]')
    
    // Click on first product to select it
    const firstProduct = page.locator('[data-testid^="product-row-"]').first()
    await firstProduct.click()
    
    // Test 'A' key for approve
    await page.keyboard.press('a')
    await page.waitForTimeout(500)
    
    // Should show approved status
    await expect(page.locator('[data-testid="product-status"]')).toContainText('approved')
    
    // Navigate to next product with arrow key
    await page.keyboard.press('ArrowDown')
    await page.waitForTimeout(500)
    
    // Test 'R' key for reject
    await page.keyboard.press('r')
    await page.waitForTimeout(500)
    
    // Should show rejected status
    await expect(page.locator('[data-testid="product-status"]')).toContainText('rejected')
  })

  test('should navigate between products', async ({ page }) => {
    // Wait for products to load
    await page.waitForSelector('[data-testid^="product-row-"]')
    
    // Get first product ID
    const firstProduct = page.locator('[data-testid^="product-row-"]').first()
    const firstProductId = await firstProduct.getAttribute('data-testid')
    
    // Click on first product
    await firstProduct.click()
    
    // Use arrow down to navigate
    await page.keyboard.press('ArrowDown')
    await page.waitForTimeout(500)
    
    // Check if selection moved to second product
    const secondProduct = page.locator('[data-testid^="product-row-"]').nth(1)
    await expect(secondProduct).toHaveClass(/selected|active/)
    
    // Use arrow up to go back
    await page.keyboard.press('ArrowUp')
    await page.waitForTimeout(500)
    
    // Should be back to first product
    await expect(firstProduct).toHaveClass(/selected|active/)
  })

  test('should show product details in side panel', async ({ page }) => {
    // Wait for products to load
    await page.waitForSelector('[data-testid^="product-row-"]')
    
    // Click on first product
    const firstProduct = page.locator('[data-testid^="product-row-"]').first()
    await firstProduct.click()
    
    // Side panel should be visible
    await expect(page.locator('[data-testid="product-detail-panel"]')).toBeVisible()
    
    // Should show product information
    await expect(page.locator('[data-testid="product-name"]')).toBeVisible()
    await expect(page.locator('[data-testid="product-description"]')).toBeVisible()
    await expect(page.locator('[data-testid="similarity-score"]')).toBeVisible()
  })

  test('should filter products by status', async ({ page }) => {
    // Wait for products to load
    await page.waitForSelector('[data-testid^="product-row-"]')
    
    // Get initial product count
    const allProducts = page.locator('[data-testid^="product-row-"]')
    const initialCount = await allProducts.count()
    
    // Filter by pending status
    await page.selectOption('[data-testid="status-filter"]', 'pending')
    await page.waitForTimeout(1000)
    
    // Should show only pending products
    const pendingProducts = page.locator('[data-testid^="product-row-"]')
    const pendingCount = await pendingProducts.count()
    
    // All visible products should have pending status
    for (let i = 0; i < pendingCount; i++) {
      const product = pendingProducts.nth(i)
      await expect(product.locator('[data-testid="status-badge"]')).toContainText('pending')
    }
    
    // Filter by approved status
    await page.selectOption('[data-testid="status-filter"]', 'approved')
    await page.waitForTimeout(1000)
    
    // Should show only approved products
    const approvedProducts = page.locator('[data-testid^="product-row-"]')
    const approvedCount = await approvedProducts.count()
    
    // Reset filter
    await page.selectOption('[data-testid="status-filter"]', 'all')
    await page.waitForTimeout(1000)
    
    // Should show all products again
    const resetProducts = page.locator('[data-testid^="product-row-"]')
    const resetCount = await resetProducts.count()
    expect(resetCount).toBe(initialCount)
  })

  test('should search products', async ({ page }) => {
    // Wait for products to load
    await page.waitForSelector('[data-testid^="product-row-"]')
    
    // Enter search term
    const searchInput = page.locator('[data-testid="search-input"]')
    await searchInput.fill('iPhone')
    
    // Wait for search results
    await page.waitForTimeout(1000)
    
    // Check if search results contain the search term
    const searchResults = page.locator('[data-testid^="product-row-"]')
    const resultCount = await searchResults.count()
    
    if (resultCount > 0) {
      // At least one result should contain "iPhone"
      const firstResult = searchResults.first()
      const productName = await firstResult.locator('[data-testid="product-name"]').textContent()
      expect(productName?.toLowerCase()).toContain('iphone')
    }
    
    // Clear search
    await searchInput.clear()
    await page.waitForTimeout(1000)
    
    // Should show all products again
    const allProducts = page.locator('[data-testid^="product-row-"]')
    const allCount = await allProducts.count()
    expect(allCount).toBeGreaterThanOrEqual(resultCount)
  })

  test('should handle batch operations', async ({ page }) => {
    // Wait for products to load
    await page.waitForSelector('[data-testid^="product-row-"]')
    
    // Select multiple products using checkboxes
    const checkboxes = page.locator('[data-testid^="product-checkbox-"]')
    const checkboxCount = await checkboxes.count()
    
    if (checkboxCount >= 2) {
      // Select first two products
      await checkboxes.first().check()
      await checkboxes.nth(1).check()
      
      // Batch actions should be visible
      await expect(page.locator('[data-testid="batch-actions"]')).toBeVisible()
      
      // Click batch approve
      await page.click('[data-testid="batch-approve-btn"]')
      
      // Wait for batch operation to complete
      await page.waitForTimeout(2000)
      
      // Should show success notification
      await expect(page.locator('.toast')).toContainText('approved')
      
      // Selected products should have approved status
      const firstProduct = page.locator('[data-testid^="product-row-"]').first()
      const secondProduct = page.locator('[data-testid^="product-row-"]').nth(1)
      
      await expect(firstProduct.locator('[data-testid="status-badge"]')).toContainText('approved')
      await expect(secondProduct.locator('[data-testid="status-badge"]')).toContainText('approved')
    }
  })

  test('should show similarity scores', async ({ page }) => {
    // Wait for products to load
    await page.waitForSelector('[data-testid^="product-row-"]')
    
    // Click on first product
    const firstProduct = page.locator('[data-testid^="product-row-"]').first()
    await firstProduct.click()
    
    // Similarity score should be visible in detail panel
    await expect(page.locator('[data-testid="similarity-score"]')).toBeVisible()
    
    // Score should be a number between 0 and 1
    const scoreText = await page.locator('[data-testid="similarity-score"]').textContent()
    const score = parseFloat(scoreText?.replace(/[^\d.]/g, '') || '0')
    expect(score).toBeGreaterThanOrEqual(0)
    expect(score).toBeLessThanOrEqual(1)
    
    // Should show similarity indicator (color coding)
    const similarityIndicator = page.locator('[data-testid="similarity-indicator"]')
    await expect(similarityIndicator).toBeVisible()
  })

  test('should handle pagination', async ({ page }) => {
    // Wait for products to load
    await page.waitForSelector('[data-testid^="product-row-"]')
    
    // Check if pagination is visible (if there are enough products)
    const pagination = page.locator('[data-testid="pagination"]')
    
    if (await pagination.isVisible()) {
      // Get current page number
      const currentPage = await page.locator('[data-testid="current-page"]').textContent()
      
      // Click next page
      await page.click('[data-testid="next-page-btn"]')
      await page.waitForTimeout(1000)
      
      // Page number should change
      const newPage = await page.locator('[data-testid="current-page"]').textContent()
      expect(newPage).not.toBe(currentPage)
      
      // Should load new products
      await expect(page.locator('[data-testid^="product-row-"]').first()).toBeVisible()
      
      // Go back to previous page
      await page.click('[data-testid="prev-page-btn"]')
      await page.waitForTimeout(1000)
      
      // Should be back to original page
      const backPage = await page.locator('[data-testid="current-page"]').textContent()
      expect(backPage).toBe(currentPage)
    }
  })

  test('should close detail panel with escape key', async ({ page }) => {
    // Wait for products to load
    await page.waitForSelector('[data-testid^="product-row-"]')
    
    // Click on first product to open detail panel
    const firstProduct = page.locator('[data-testid^="product-row-"]').first()
    await firstProduct.click()
    
    // Detail panel should be visible
    await expect(page.locator('[data-testid="product-detail-panel"]')).toBeVisible()
    
    // Press escape key
    await page.keyboard.press('Escape')
    await page.waitForTimeout(500)
    
    // Detail panel should be hidden
    await expect(page.locator('[data-testid="product-detail-panel"]')).not.toBeVisible()
  })

  test('should handle loading and error states', async ({ page }) => {
    // Navigate to products page
    await page.goto('/products')
    
    // Should show loading indicator initially
    await expect(page.locator('[data-testid="loading-indicator"]')).toBeVisible()
    
    // Wait for data to load
    await page.waitForLoadState('networkidle')
    
    // Loading indicator should disappear
    await expect(page.locator('[data-testid="loading-indicator"]')).not.toBeVisible()
    
    // Mock network error for next request
    await page.route('**/api/products**', route => {
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Server error' })
      })
    })
    
    // Trigger a refresh or new request
    await page.reload()
    await page.waitForTimeout(2000)
    
    // Should show error message
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible()
  })
})
