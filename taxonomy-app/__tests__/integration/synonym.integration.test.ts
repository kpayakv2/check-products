import { 
  setupTestDatabase, 
  clearTestData, 
  testSupabase,
  getTestSynonyms 
} from '../setup/database-setup'

describe('Synonym Integration Tests', () => {
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

  describe('Synonym CRUD Operations', () => {
    it('should create synonym with terms', async () => {
      const newSynonym = {
        id: 'test-syn-new',
        lemma: 'test-laptop',
        category_id: 'test-tax-2',
        confidence_score: 0.88,
        is_verified: false,
      }

      // Create synonym
      const { data: synonymData, error: synonymError } = await testSupabase
        .from('synonyms')
        .insert([newSynonym])
        .select()
        .single()

      expect(synonymError).toBeNull()
      expect(synonymData.lemma).toBe('test-laptop')

      // Add terms
      const terms = [
        {
          id: 'test-term-new-1',
          synonym_id: 'test-syn-new',
          term: 'notebook computer',
          language: 'en',
        },
        {
          id: 'test-term-new-2',
          synonym_id: 'test-syn-new',
          term: 'แล็ปท็อป',
          language: 'th',
        }
      ]

      const { error: termError } = await testSupabase
        .from('synonym_terms')
        .insert(terms)

      expect(termError).toBeNull()

      // Verify with join
      const synonyms = await getTestSynonyms()
      const newSyn = synonyms.find(s => s.id === 'test-syn-new')
      
      expect(newSyn).toBeDefined()
      expect(newSyn?.synonym_terms).toHaveLength(2)
    })

    it('should read synonyms with terms', async () => {
      const synonyms = await getTestSynonyms()

      expect(synonyms).toHaveLength(1)
      
      const synonym = synonyms[0]
      expect(synonym.lemma).toBe('test-smartphone')
      expect(synonym.confidence_score).toBe(0.95)
      expect(synonym.synonym_terms).toHaveLength(2)

      // Check terms
      const englishTerm = synonym.synonym_terms.find(t => t.language === 'en')
      const thaiTerm = synonym.synonym_terms.find(t => t.language === 'th')

      expect(englishTerm?.term).toBe('test smart phone')
      expect(thaiTerm?.term).toBe('เทสมือถือ')
    })

    it('should update synonym confidence score', async () => {
      const { data, error } = await testSupabase
        .from('synonyms')
        .update({ 
          confidence_score: 0.99,
          is_verified: true 
        })
        .eq('id', 'test-syn-1')
        .select()
        .single()

      expect(error).toBeNull()
      expect(data.confidence_score).toBe(0.99)
      expect(data.is_verified).toBe(true)
    })

    it('should delete synonym and cascade terms', async () => {
      // Delete synonym (should cascade to terms)
      const { error } = await testSupabase
        .from('synonyms')
        .delete()
        .eq('id', 'test-syn-1')

      expect(error).toBeNull()

      // Verify synonym deleted
      const { data: synonymData } = await testSupabase
        .from('synonyms')
        .select('*')
        .eq('id', 'test-syn-1')

      expect(synonymData).toHaveLength(0)

      // Verify terms deleted (cascade)
      const { data: termData } = await testSupabase
        .from('synonym_terms')
        .select('*')
        .eq('synonym_id', 'test-syn-1')

      expect(termData).toHaveLength(0)
    })
  })

  describe('Synonym Business Logic', () => {
    it('should enforce confidence score range', async () => {
      const invalidSynonym = {
        id: 'test-invalid-1',
        lemma: 'invalid-confidence',
        category_id: 'test-tax-2',
        confidence_score: 1.5, // Invalid > 1
        is_verified: false,
      }

      const { error } = await testSupabase
        .from('synonyms')
        .insert([invalidSynonym])

      // Should fail due to check constraint
      expect(error).not.toBeNull()
    })

    it('should handle multilingual terms', async () => {
      // Add more language variants
      const additionalTerms = [
        {
          id: 'test-term-3',
          synonym_id: 'test-syn-1',
          term: 'teléfono inteligente',
          language: 'es',
        },
        {
          id: 'test-term-4',
          synonym_id: 'test-syn-1',
          term: 'スマートフォン',
          language: 'ja',
        }
      ]

      const { error } = await testSupabase
        .from('synonym_terms')
        .insert(additionalTerms)

      expect(error).toBeNull()

      // Verify all languages
      const { data } = await testSupabase
        .from('synonym_terms')
        .select('*')
        .eq('synonym_id', 'test-syn-1')

      expect(data).toHaveLength(4)
      
      const languages = data!.map(t => t.language)
      expect(languages).toContain('en')
      expect(languages).toContain('th')
      expect(languages).toContain('es')
      expect(languages).toContain('ja')
    })

    it('should link to valid taxonomy category', async () => {
      const invalidSynonym = {
        id: 'test-invalid-2',
        lemma: 'invalid-category',
        category_id: 'non-existent-category',
        confidence_score: 0.5,
        is_verified: false,
      }

      const { error } = await testSupabase
        .from('synonyms')
        .insert([invalidSynonym])

      // Should fail due to foreign key constraint
      expect(error).not.toBeNull()
      expect(error?.code).toBe('23503') // PostgreSQL foreign key violation
    })
  })

  describe('Synonym Search and Analytics', () => {
    it('should search synonyms by lemma', async () => {
      const { data, error } = await testSupabase
        .from('synonyms')
        .select('*')
        .ilike('lemma', '%smartphone%')

      expect(error).toBeNull()
      expect(data).toHaveLength(1)
      expect(data![0].lemma).toContain('smartphone')
    })

    it('should filter by confidence score range', async () => {
      // Add synonym with different confidence
      await testSupabase
        .from('synonyms')
        .insert([{
          id: 'test-syn-low',
          lemma: 'low-confidence',
          category_id: 'test-tax-2',
          confidence_score: 0.3,
          is_verified: false,
        }])

      const { data, error } = await testSupabase
        .from('synonyms')
        .select('*')
        .gte('confidence_score', 0.8)

      expect(error).toBeNull()
      expect(data).toHaveLength(1)
      expect(data![0].confidence_score).toBeGreaterThanOrEqual(0.8)
    })

    it('should get synonyms by verification status', async () => {
      const { data, error } = await testSupabase
        .from('synonyms')
        .select('*')
        .eq('is_verified', true)

      expect(error).toBeNull()
      expect(data).toHaveLength(1)
      expect(data![0].is_verified).toBe(true)
    })

    it('should search terms across languages', async () => {
      const { data, error } = await testSupabase
        .from('synonym_terms')
        .select(`
          *,
          synonyms (lemma, category_id)
        `)
        .ilike('term', '%มือถือ%')

      expect(error).toBeNull()
      expect(data).toHaveLength(1)
      expect(data![0].term).toContain('มือถือ')
      expect(data![0].language).toBe('th')
    })
  })
})
