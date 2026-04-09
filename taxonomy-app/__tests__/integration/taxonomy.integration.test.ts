import { 
  setupTestDatabase, 
  clearTestData, 
  testSupabase,
  getTestTaxonomyNodes,
  seedTaxonomyData 
} from '../setup/database-setup'

describe('Taxonomy Integration Tests', () => {
  // Setup และ cleanup สำหรับแต่ละ test
  beforeAll(async () => {
    await setupTestDatabase()
  })

  afterAll(async () => {
    await clearTestData()
  })

  beforeEach(async () => {
    // Reset data ก่อนแต่ละ test
    await clearTestData()
    await setupTestDatabase()
  })

  describe('Taxonomy CRUD Operations', () => {
    it('should create new taxonomy node', async () => {
      const newNode = {
        id: 'test-new-1',
        name: 'Test New Category',
        code: 'TESTNEW001',
        parent_id: 'test-tax-1',
        level: 2,
        sort_order: 2,
        is_active: true,
      }

      const { data, error } = await testSupabase
        .from('taxonomy_nodes')
        .insert([newNode])
        .select()
        .single()

      expect(error).toBeNull()
      expect(data).toBeDefined()
      expect(data.name).toBe('Test New Category')
      expect(data.code).toBe('TESTNEW001')
    })

    it('should read taxonomy nodes with hierarchy', async () => {
      const nodes = await getTestTaxonomyNodes()

      expect(nodes).toHaveLength(2)
      
      const rootNode = nodes.find(n => n.parent_id === null)
      const childNode = nodes.find(n => n.parent_id !== null)

      expect(rootNode).toBeDefined()
      expect(rootNode?.level).toBe(1)
      expect(rootNode?.name).toBe('Test Electronics')

      expect(childNode).toBeDefined()
      expect(childNode?.level).toBe(2)
      expect(childNode?.parent_id).toBe(rootNode?.id)
    })

    it('should update taxonomy node', async () => {
      const { data, error } = await testSupabase
        .from('taxonomy_nodes')
        .update({ name: 'Updated Test Electronics' })
        .eq('id', 'test-tax-1')
        .select()
        .single()

      expect(error).toBeNull()
      expect(data.name).toBe('Updated Test Electronics')
    })

    it('should delete taxonomy node', async () => {
      // Delete child first (foreign key constraint)
      await testSupabase
        .from('taxonomy_nodes')
        .delete()
        .eq('id', 'test-tax-2')

      const { error } = await testSupabase
        .from('taxonomy_nodes')
        .delete()
        .eq('id', 'test-tax-1')

      expect(error).toBeNull()

      // Verify deletion
      const { data } = await testSupabase
        .from('taxonomy_nodes')
        .select('*')
        .eq('id', 'test-tax-1')

      expect(data).toHaveLength(0)
    })
  })

  describe('Taxonomy Business Logic', () => {
    it('should enforce unique codes', async () => {
      const duplicateNode = {
        id: 'test-dup-1',
        name: 'Duplicate Code Test',
        code: 'TEST001', // Same as existing
        parent_id: null,
        level: 1,
        sort_order: 2,
        is_active: true,
      }

      const { error } = await testSupabase
        .from('taxonomy_nodes')
        .insert([duplicateNode])

      // Should fail due to unique constraint
      expect(error).not.toBeNull()
      expect(error?.code).toBe('23505') // PostgreSQL unique violation
    })

    it('should maintain proper hierarchy levels', async () => {
      const nodes = await getTestTaxonomyNodes()
      
      // Check level consistency
      nodes.forEach(node => {
        if (node.parent_id === null) {
          expect(node.level).toBe(1)
        } else {
          const parent = nodes.find(n => n.id === node.parent_id)
          expect(node.level).toBe(parent!.level + 1)
        }
      })
    })

    it('should handle sort order correctly', async () => {
      // Add multiple nodes at same level
      const newNodes = [
        {
          id: 'test-sort-1',
          name: 'Sort Test 1',
          code: 'SORT001',
          parent_id: 'test-tax-1',
          level: 2,
          sort_order: 1,
          is_active: true,
        },
        {
          id: 'test-sort-2',
          name: 'Sort Test 2', 
          code: 'SORT002',
          parent_id: 'test-tax-1',
          level: 2,
          sort_order: 2,
          is_active: true,
        }
      ]

      const { error } = await testSupabase
        .from('taxonomy_nodes')
        .insert(newNodes)

      expect(error).toBeNull()

      // Verify sort order
      const { data } = await testSupabase
        .from('taxonomy_nodes')
        .select('*')
        .eq('parent_id', 'test-tax-1')
        .order('sort_order', { ascending: true })

      expect(data).toHaveLength(3) // Including existing test-tax-2
      expect(data![0].sort_order).toBeLessThanOrEqual(data![1].sort_order)
    })
  })

  describe('Taxonomy Search and Filtering', () => {
    it('should search by name', async () => {
      const { data, error } = await testSupabase
        .from('taxonomy_nodes')
        .select('*')
        .ilike('name', '%electronics%')

      expect(error).toBeNull()
      expect(data).toHaveLength(1)
      expect(data![0].name).toContain('Electronics')
    })

    it('should filter by level', async () => {
      const { data, error } = await testSupabase
        .from('taxonomy_nodes')
        .select('*')
        .eq('level', 2)

      expect(error).toBeNull()
      expect(data).toHaveLength(1)
      expect(data![0].level).toBe(2)
    })

    it('should filter by active status', async () => {
      // Deactivate one node
      await testSupabase
        .from('taxonomy_nodes')
        .update({ is_active: false })
        .eq('id', 'test-tax-2')

      const { data, error } = await testSupabase
        .from('taxonomy_nodes')
        .select('*')
        .eq('is_active', true)

      expect(error).toBeNull()
      expect(data).toHaveLength(1)
      expect(data![0].is_active).toBe(true)
    })
  })
})
