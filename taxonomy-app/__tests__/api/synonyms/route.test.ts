import { generateSynonymTestData } from '../../utils/api-test-utils'

// Simple Synonyms API tests
describe('Synonyms API Route', () => {
  it('should generate synonym test data', () => {
    const synonyms = generateSynonymTestData()
    expect(synonyms).toHaveLength(1)
    expect(synonyms[0].lemma).toBe('smartphone')
  })

  it('should validate synonym data structure', () => {
    const synonyms = generateSynonymTestData()
    
    synonyms.forEach(synonym => {
      expect(synonym).toHaveProperty('id')
      expect(synonym).toHaveProperty('lemma')
      expect(synonym).toHaveProperty('category_id')
      expect(synonym).toHaveProperty('confidence_score')
      expect(synonym).toHaveProperty('is_verified')
      expect(synonym).toHaveProperty('terms')
    })
  })

  it('should handle synonym terms', () => {
    const synonyms = generateSynonymTestData()
    const synonym = synonyms[0]
    
    expect(synonym.terms).toHaveLength(3)
    expect(synonym.terms[0].term).toBe('smart phone')
    expect(synonym.terms[1].term).toBe('มือถือ')
  })

  it('should validate confidence scores', () => {
    const synonyms = generateSynonymTestData()
    
    synonyms.forEach(synonym => {
      expect(synonym.confidence_score).toBeGreaterThan(0)
      expect(synonym.confidence_score).toBeLessThanOrEqual(1)
    })
  })

  it('should handle multi-language terms', () => {
    const synonyms = generateSynonymTestData()
    const terms = synonyms[0].terms
    
    const englishTerms = terms.filter(t => t.language === 'en')
    const thaiTerms = terms.filter(t => t.language === 'th')
    
    expect(englishTerms).toHaveLength(1)
    expect(thaiTerms).toHaveLength(2)
  })
})
