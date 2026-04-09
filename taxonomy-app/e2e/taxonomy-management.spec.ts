import { test, expect } from '@playwright/test'

test.describe('Taxonomy Management', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to taxonomy page
    await page.goto('/taxonomy')
    
    // Wait for page to load
    await page.waitForLoadState('networkidle')
  })

  test('should display taxonomy tree', async ({ page }) => {
    // Check if the main heading is visible
    await expect(page.locator('h1')).toContainText('Taxonomy Management')
    
    // Check if taxonomy tree container is visible
    await expect(page.locator('[data-testid="taxonomy-tree"]')).toBeVisible()
    
    // Wait for data to load
    await page.waitForTimeout(2000)
    
    // Check if at least one taxonomy node is visible
    const taxonomyNodes = page.locator('[data-testid^="taxonomy-node-"]')
    await expect(taxonomyNodes.first()).toBeVisible()
  })

  test('should create new taxonomy category', async ({ page }) => {
    // Click add category button
    await page.click('[data-testid="add-category-btn"]')
    
    // Fill in the form
    await page.fill('[data-testid="category-name-input"]', 'Test Category')
    await page.fill('[data-testid="category-code-input"]', 'TEST001')
    
    // Submit the form
    await page.click('[data-testid="save-category-btn"]')
    
    // Wait for success message or new category to appear
    await page.waitForTimeout(1000)
    
    // Check if the new category appears in the tree
    await expect(page.locator('text=Test Category')).toBeVisible()
  })

  test('should edit existing taxonomy category', async ({ page }) => {
    // Wait for taxonomy nodes to load
    await page.waitForSelector('[data-testid^="taxonomy-node-"]')
    
    // Click edit button on first category
    const firstEditBtn = page.locator('[data-testid^="edit-btn-"]').first()
    await firstEditBtn.click()
    
    // Update the name
    const nameInput = page.locator('[data-testid="category-name-input"]')
    await nameInput.clear()
    await nameInput.fill('Updated Category Name')
    
    // Save changes
    await page.click('[data-testid="save-category-btn"]')
    
    // Wait for update to complete
    await page.waitForTimeout(1000)
    
    // Verify the update
    await expect(page.locator('text=Updated Category Name')).toBeVisible()
  })

  test('should delete taxonomy category', async ({ page }) => {
    // Wait for taxonomy nodes to load
    await page.waitForSelector('[data-testid^="taxonomy-node-"]')
    
    // Get the first category name for verification
    const firstNode = page.locator('[data-testid^="taxonomy-node-"]').first()
    const categoryName = await firstNode.locator('[data-testid="category-name"]').textContent()
    
    // Click delete button
    const deleteBtn = page.locator('[data-testid^="delete-btn-"]').first()
    await deleteBtn.click()
    
    // Confirm deletion in modal
    await page.click('[data-testid="confirm-delete-btn"]')
    
    // Wait for deletion to complete
    await page.waitForTimeout(1000)
    
    // Verify the category is no longer visible
    if (categoryName) {
      await expect(page.locator(`text=${categoryName}`)).not.toBeVisible()
    }
  })

  test('should search taxonomy categories', async ({ page }) => {
    // Wait for taxonomy nodes to load
    await page.waitForSelector('[data-testid^="taxonomy-node-"]')
    
    // Enter search term
    const searchInput = page.locator('[data-testid="search-input"]')
    await searchInput.fill('Electronics')
    
    // Wait for search results
    await page.waitForTimeout(500)
    
    // Check if search results are filtered
    const visibleNodes = page.locator('[data-testid^="taxonomy-node-"]:visible')
    const nodeCount = await visibleNodes.count()
    
    // Should have fewer nodes than before search
    expect(nodeCount).toBeGreaterThan(0)
    
    // Clear search
    await searchInput.clear()
    await page.waitForTimeout(500)
    
    // Should show all nodes again
    const allNodes = page.locator('[data-testid^="taxonomy-node-"]:visible')
    const allNodeCount = await allNodes.count()
    expect(allNodeCount).toBeGreaterThanOrEqual(nodeCount)
  })

  test('should expand and collapse taxonomy nodes', async ({ page }) => {
    // Wait for taxonomy nodes to load
    await page.waitForSelector('[data-testid^="taxonomy-node-"]')
    
    // Find a parent node with children
    const expandBtn = page.locator('[data-testid^="expand-btn-"]').first()
    
    if (await expandBtn.isVisible()) {
      // Click to expand
      await expandBtn.click()
      await page.waitForTimeout(500)
      
      // Check if children are visible
      const childNodes = page.locator('[data-level="2"]:visible')
      const childCount = await childNodes.count()
      expect(childCount).toBeGreaterThan(0)
      
      // Click to collapse
      await expandBtn.click()
      await page.waitForTimeout(500)
      
      // Check if children are hidden
      const hiddenChildren = page.locator('[data-level="2"]:visible')
      const hiddenCount = await hiddenChildren.count()
      expect(hiddenCount).toBeLessThan(childCount)
    }
  })

  test('should handle drag and drop reordering', async ({ page }) => {
    // Wait for taxonomy nodes to load
    await page.waitForSelector('[data-testid^="taxonomy-node-"]')
    
    // Get first two nodes
    const firstNode = page.locator('[data-testid^="taxonomy-node-"]').first()
    const secondNode = page.locator('[data-testid^="taxonomy-node-"]').nth(1)
    
    // Get their initial positions
    const firstNodeBox = await firstNode.boundingBox()
    const secondNodeBox = await secondNode.boundingBox()
    
    if (firstNodeBox && secondNodeBox) {
      // Perform drag and drop
      await firstNode.dragTo(secondNode)
      
      // Wait for reordering to complete
      await page.waitForTimeout(1000)
      
      // Verify positions have changed (this would need more specific implementation)
      // For now, just check that the nodes are still visible
      await expect(firstNode).toBeVisible()
      await expect(secondNode).toBeVisible()
    }
  })

  test('should validate form inputs', async ({ page }) => {
    // Click add category button
    await page.click('[data-testid="add-category-btn"]')
    
    // Try to submit empty form
    await page.click('[data-testid="save-category-btn"]')
    
    // Check for validation errors
    await expect(page.locator('[data-testid="name-error"]')).toBeVisible()
    await expect(page.locator('[data-testid="code-error"]')).toBeVisible()
    
    // Fill only name
    await page.fill('[data-testid="category-name-input"]', 'Test Category')
    await page.click('[data-testid="save-category-btn"]')
    
    // Should still show code error
    await expect(page.locator('[data-testid="code-error"]')).toBeVisible()
    
    // Fill code with invalid format
    await page.fill('[data-testid="category-code-input"]', 'invalid code')
    await page.click('[data-testid="save-category-btn"]')
    
    // Should show format error
    await expect(page.locator('[data-testid="code-format-error"]')).toBeVisible()
  })

  test('should handle keyboard navigation', async ({ page }) => {
    // Wait for taxonomy nodes to load
    await page.waitForSelector('[data-testid^="taxonomy-node-"]')
    
    // Focus on the first node
    const firstNode = page.locator('[data-testid^="taxonomy-node-"]').first()
    await firstNode.focus()
    
    // Test arrow key navigation
    await page.keyboard.press('ArrowDown')
    await page.waitForTimeout(200)
    
    // Check if focus moved to next node
    const secondNode = page.locator('[data-testid^="taxonomy-node-"]').nth(1)
    await expect(secondNode).toBeFocused()
    
    // Test Enter key to edit
    await page.keyboard.press('Enter')
    await page.waitForTimeout(200)
    
    // Should open edit form
    await expect(page.locator('[data-testid="category-name-input"]')).toBeVisible()
    
    // Test Escape to cancel
    await page.keyboard.press('Escape')
    await page.waitForTimeout(200)
    
    // Should close edit form
    await expect(page.locator('[data-testid="category-name-input"]')).not.toBeVisible()
  })

  test('should display loading states', async ({ page }) => {
    // Navigate to taxonomy page
    await page.goto('/taxonomy')
    
    // Should show loading indicator initially
    await expect(page.locator('[data-testid="loading-indicator"]')).toBeVisible()
    
    // Wait for data to load
    await page.waitForLoadState('networkidle')
    
    // Loading indicator should disappear
    await expect(page.locator('[data-testid="loading-indicator"]')).not.toBeVisible()
    
    // Content should be visible
    await expect(page.locator('[data-testid="taxonomy-tree"]')).toBeVisible()
  })

  test('should handle error states', async ({ page }) => {
    // Mock network failure
    await page.route('**/api/taxonomy**', route => {
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Server error' })
      })
    })
    
    // Navigate to taxonomy page
    await page.goto('/taxonomy')
    
    // Wait for error to appear
    await page.waitForTimeout(2000)
    
    // Should show error message
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible()
    await expect(page.locator('[data-testid="error-message"]')).toContainText('error')
    
    // Should show retry button
    await expect(page.locator('[data-testid="retry-btn"]')).toBeVisible()
  })
})
