import { 
  setupTestDatabase, 
  clearTestData, 
  testSupabase,
  getTestProducts 
} from '../setup/database-setup'

describe('Product Integration Tests', () => {
  beforeAll(async () => {
    await setupTestDatabase()
  })

  afterAll(async () => {
    await clearTestData()
  })

  beforeEach(async () => {
    await clearTestData()
    await setupTestDatabase()
  })

  describe('Product CRUD Operations', () => {
    it('should create new product', async () => {
      const newProduct = {
        id: 'test-prod-new',
        name: 'Test Samsung Galaxy S24',
        description: 'Test premium Android smartphone',
        category_id: 'test-tax-2',
        status: 'pending',
        similarity_score: 0.92,
      }

      const { data, error } = await testSupabase
        .from('products')
        .insert([newProduct])
        .select()
        .single()

      expect(error).toBeNull()
      expect(data.name).toBe('Test Samsung Galaxy S24')
      expect(data.status).toBe('pending')
      expect(data.similarity_score).toBe(0.92)
    })

    it('should read products with category info', async () => {
      const { data, error } = await testSupabase
        .from('products')
        .select(`
          *,
          taxonomy_nodes (
            id,
            name,
            code
          )
        `)
        .like('id', 'test-%')

      expect(error).toBeNull()
      expect(data).toHaveLength(1)
      
      const product = data![0]
      expect(product.name).toBe('Test iPhone 15 Pro')
      expect(product.taxonomy_nodes).toBeDefined()
      expect(product.taxonomy_nodes.name).toBe('Test Smartphones')
    })

    it('should update product status', async () => {
      const { data, error } = await testSupabase
        .from('products')
        .update({ 
          status: 'approved',
          updated_at: new Date().toISOString()
        })
        .eq('id', 'test-prod-1')
        .select()
        .single()

      expect(error).toBeNull()
      expect(data.status).toBe('approved')
      expect(new Date(data.updated_at)).toBeInstanceOf(Date)
    })

    it('should delete product', async () => {
      const { error } = await testSupabase
        .from('products')
        .delete()
        .eq('id', 'test-prod-1')

      expect(error).toBeNull()

      // Verify deletion
      const { data } = await testSupabase
        .from('products')
        .select('*')
        .eq('id', 'test-prod-1')

      expect(data).toHaveLength(0)
    })
  })

  describe('Product Status Management', () => {
    it('should handle status transitions', async () => {
      const statusFlow = ['pending', 'approved', 'rejected']

      for (const status of statusFlow) {
        const { data, error } = await testSupabase
          .from('products')
          .update({ status })
          .eq('id', 'test-prod-1')
          .select()
          .single()

        expect(error).toBeNull()
        expect(data.status).toBe(status)
      }
    })

    it('should reject invalid status', async () => {
      const { error } = await testSupabase
        .from('products')
        .update({ status: 'invalid-status' })
        .eq('id', 'test-prod-1')

      // Should fail due to check constraint
      expect(error).not.toBeNull()
    })

    it('should track status history', async () => {
      // Create review history entry
      const historyEntry = {
        id: 'test-history-1',
        product_id: 'test-prod-1',
        action: 'approved',
        reviewer_id: 'test-user-1',
        notes: 'Test approval',
        created_at: new Date().toISOString()
      }

      const { error } = await testSupabase
        .from('review_history')
        .insert([historyEntry])

      expect(error).toBeNull()

      // Verify history
      const { data } = await testSupabase
        .from('review_history')
        .select('*')
        .eq('product_id', 'test-prod-1')

      expect(data).toHaveLength(1)
      expect(data![0].action).toBe('approved')
    })
  })

  describe('Product Similarity and Matching', () => {
    it('should enforce similarity score range', async () => {
      const invalidProduct = {
        id: 'test-invalid-sim',
        name: 'Invalid Similarity',
        description: 'Test invalid similarity score',
        category_id: 'test-tax-2',
        status: 'pending',
        similarity_score: 1.5, // Invalid > 1
      }

      const { error } = await testSupabase
        .from('products')
        .insert([invalidProduct])

      // Should fail due to check constraint
      expect(error).not.toBeNull()
    })

    it('should create similarity matches', async () => {
      // Add another product for matching
      await testSupabase
        .from('products')
        .insert([{
          id: 'test-prod-2',
          name: 'Test iPhone 15',
          description: 'Test standard iPhone model',
          category_id: 'test-tax-2',
          status: 'pending',
          similarity_score: 0.78,
        }])

      // Create similarity match
      const similarityMatch = {
        id: 'test-match-1',
        product_a_id: 'test-prod-1',
        product_b_id: 'test-prod-2',
        similarity_score: 0.85,
        match_type: 'name_similarity',
        created_at: new Date().toISOString()
      }

      const { error } = await testSupabase
        .from('similarity_matches')
        .insert([similarityMatch])

      expect(error).toBeNull()

      // Verify match with product details
      const { data } = await testSupabase
        .from('similarity_matches')
        .select(`
          *,
          product_a:products!product_a_id(name),
          product_b:products!product_b_id(name)
        `)
        .eq('id', 'test-match-1')
        .single()

      expect(data.similarity_score).toBe(0.85)
      expect(data.product_a.name).toBe('Test iPhone 15 Pro')
      expect(data.product_b.name).toBe('Test iPhone 15')
    })

    it('should find high similarity products', async () => {
      // Add products with various similarity scores
      const products = [
        {
          id: 'test-high-sim',
          name: 'High Similarity Product',
          description: 'Test high similarity',
          category_id: 'test-tax-2',
          status: 'pending',
          similarity_score: 0.95,
        },
        {
          id: 'test-low-sim',
          name: 'Low Similarity Product',
          description: 'Test low similarity',
          category_id: 'test-tax-2',
          status: 'pending',
          similarity_score: 0.45,
        }
      ]

      await testSupabase.from('products').insert(products)

      // Query high similarity products
      const { data, error } = await testSupabase
        .from('products')
        .select('*')
        .gte('similarity_score', 0.8)
        .order('similarity_score', { ascending: false })

      expect(error).toBeNull()
      expect(data!.length).toBeGreaterThanOrEqual(2)
      expect(data![0].similarity_score).toBeGreaterThanOrEqual(0.8)
    })
  })

  describe('Product Search and Filtering', () => {
    it('should search products by name', async () => {
      const { data, error } = await testSupabase
        .from('products')
        .select('*')
        .ilike('name', '%iphone%')

      expect(error).toBeNull()
      expect(data).toHaveLength(1)
      expect(data![0].name.toLowerCase()).toContain('iphone')
    })

    it('should filter by status', async () => {
      // Add products with different statuses
      await testSupabase.from('products').insert([
        {
          id: 'test-approved',
          name: 'Approved Product',
          description: 'Test approved product',
          category_id: 'test-tax-2',
          status: 'approved',
          similarity_score: 0.8,
        },
        {
          id: 'test-rejected',
          name: 'Rejected Product',
          description: 'Test rejected product',
          category_id: 'test-tax-2',
          status: 'rejected',
          similarity_score: 0.6,
        }
      ])

      // Filter by status
      const { data: pendingData } = await testSupabase
        .from('products')
        .select('*')
        .eq('status', 'pending')

      const { data: approvedData } = await testSupabase
        .from('products')
        .select('*')
        .eq('status', 'approved')

      expect(pendingData).toHaveLength(1)
      expect(approvedData).toHaveLength(1)
    })

    it('should filter by category', async () => {
      const { data, error } = await testSupabase
        .from('products')
        .select('*')
        .eq('category_id', 'test-tax-2')

      expect(error).toBeNull()
      expect(data).toHaveLength(1)
      expect(data![0].category_id).toBe('test-tax-2')
    })

    it('should paginate results', async () => {
      // Add more products for pagination test
      const moreProducts = Array.from({ length: 15 }, (_, i) => ({
        id: `test-page-${i}`,
        name: `Page Test Product ${i}`,
        description: `Test product for pagination ${i}`,
        category_id: 'test-tax-2',
        status: 'pending',
        similarity_score: 0.5 + (i * 0.01),
      }))

      await testSupabase.from('products').insert(moreProducts)

      // Test pagination
      const { data: page1, error } = await testSupabase
        .from('products')
        .select('*')
        .like('id', 'test-%')
        .order('created_at', { ascending: false })
        .range(0, 9) // First 10 items

      expect(error).toBeNull()
      expect(page1).toHaveLength(10)

      const { data: page2 } = await testSupabase
        .from('products')
        .select('*')
        .like('id', 'test-%')
        .order('created_at', { ascending: false })
        .range(10, 19) // Next 10 items

      expect(page2!.length).toBeGreaterThan(0)
      expect(page1![0].id).not.toBe(page2![0].id)
    })
  })
})
