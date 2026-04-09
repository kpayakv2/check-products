import { generateTaxonomyTestData } from '../../utils/api-test-utils'

// Simple Taxonomy API tests
describe('Taxonomy API Route', () => {
  it('should generate taxonomy test data', () => {
    const taxonomy = generateTaxonomyTestData()
    expect(taxonomy).toHaveLength(2)
    expect(taxonomy[0].name).toBe('Electronics')
    expect(taxonomy[1].name).toBe('Smartphones')
  })

  it('should validate taxonomy data structure', () => {
    const taxonomy = generateTaxonomyTestData()
    
    taxonomy.forEach(node => {
      expect(node).toHaveProperty('id')
      expect(node).toHaveProperty('name')
      expect(node).toHaveProperty('code')
      expect(node).toHaveProperty('level')
      expect(node).toHaveProperty('sort_order')
      expect(node).toHaveProperty('is_active')
    })
  })

  it('should handle hierarchical structure', () => {
    const taxonomy = generateTaxonomyTestData()
    const rootNode = taxonomy.find(n => n.parent_id === null)
    const childNode = taxonomy.find(n => n.parent_id !== null)
    
    expect(rootNode).toBeDefined()
    expect(childNode).toBeDefined()
    expect(rootNode?.level).toBe(1)
    expect(childNode?.level).toBe(2)
  })

  it('should validate taxonomy codes', () => {
    const taxonomy = generateTaxonomyTestData()
    
    taxonomy.forEach(node => {
      expect(node.code).toMatch(/^[A-Z]+\d+$/)
    })
  })

  it('should handle active status', () => {
    const taxonomy = generateTaxonomyTestData()
    
    taxonomy.forEach(node => {
      expect(typeof node.is_active).toBe('boolean')
    })
  })
})
