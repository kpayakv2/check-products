import { supabase } from './supabase'

export class DatabaseService {
  // Taxonomy Operations
  static async getTaxonomyTree() {
    try {
      const { data, error } = await supabase
        .from('taxonomy_nodes')
        .select('*')
        .order('sort_order')
      
      if (error) throw error
      return data
    } catch (error) {
      console.error('Error fetching taxonomy tree:', error)
      throw error
    }
  }

  static async createCategory(categoryData: any) {
    try {
      const { data, error } = await supabase
        .from('taxonomy_nodes')
        .insert([categoryData])
        .select()
      
      if (error) throw error
      return data[0]
    } catch (error) {
      console.error('Error creating category:', error)
      throw error
    }
  }

  static async updateCategory(id: string, categoryData: any) {
    try {
      const { data, error } = await supabase
        .from('taxonomy_nodes')
        .update(categoryData)
        .eq('id', id)
        .select()
      
      if (error) throw error
      return data[0]
    } catch (error) {
      console.error('Error updating category:', error)
      throw error
    }
  }

  static async deleteCategory(id: string) {
    try {
      const { error } = await supabase
        .from('taxonomy_nodes')
        .delete()
        .eq('id', id)
      
      if (error) throw error
      return true
    } catch (error) {
      console.error('Error deleting category:', error)
      throw error
    }
  }

  // Synonym Operations - Using normalized schema (synonym_lemmas + synonym_terms)
  static async getSynonyms() {
    try {
      const { data, error } = await supabase
        .from('synonym_lemmas')
        .select(`
          *,
          synonym_terms (
            id,
            term,
            is_primary,
            confidence_score,
            is_verified,
            source,
            created_at
          )
        `)
        .eq('is_active', true)
        .order('created_at', { ascending: false })
      
      if (error) throw error
      
      // Transform to flat structure for frontend compatibility
      const flatSynonyms: any[] = []
      data?.forEach((lemma: any) => {
        lemma.synonym_terms?.forEach((term: any) => {
          flatSynonyms.push({
            id: term.id,
            main_term: lemma.name_th,
            synonym_term: term.term,
            category_id: lemma.category_id,
            confidence_score: term.confidence_score || 0.0,
            is_verified: term.is_verified || false,
            source: term.source || 'manual',
            created_at: term.created_at,
            updated_at: lemma.updated_at,
            // Additional fields for reference
            lemma_id: lemma.id,
            lemma_name_en: lemma.name_en,
            is_primary: term.is_primary
          })
        })
      })
      
      return flatSynonyms
    } catch (error) {
      console.error('Error fetching synonyms:', error)
      throw error
    }
  }

  static async createSynonym(synonymData: any) {
    try {
      // Create or find lemma first
      let lemmaId = null
      
      // Check if lemma exists
      const { data: existingLemma, error: lemmaError } = await supabase
        .from('synonym_lemmas')
        .select('id')
        .eq('name_th', synonymData.main_term)
        .single()
      
      if (existingLemma) {
        lemmaId = existingLemma.id
      } else {
        // Create new lemma
        const { data: newLemma, error: createLemmaError } = await supabase
          .from('synonym_lemmas')
          .insert([{
            name_th: synonymData.main_term,
            category_id: synonymData.category_id,
            is_verified: synonymData.is_verified || false,
            is_active: true
          }])
          .select()
          .single()
        
        if (createLemmaError) throw createLemmaError
        lemmaId = newLemma.id
      }
      
      // Create synonym term
      const { data, error } = await supabase
        .from('synonym_terms')
        .insert([{
          lemma_id: lemmaId,
          term: synonymData.synonym_term,
          confidence_score: synonymData.confidence_score || 0.0,
          is_verified: synonymData.is_verified || false,
          source: synonymData.source || 'manual'
        }])
        .select()
      
      if (error) throw error
      return data[0]
    } catch (error) {
      console.error('Error creating synonym:', error)
      throw error
    }
  }

  static async updateSynonym(id: string, synonymData: any) {
    try {
      // Update synonym term
      const { data, error } = await supabase
        .from('synonym_terms')
        .update({
          term: synonymData.synonym_term,
          confidence_score: synonymData.confidence_score,
          is_verified: synonymData.is_verified,
          source: synonymData.source
        })
        .eq('id', id)
        .select()
      
      if (error) throw error
      return data[0]
    } catch (error) {
      console.error('Error updating synonym:', error)
      throw error
    }
  }

  static async deleteSynonym(id: string) {
    try {
      // Delete synonym term
      const { error } = await supabase
        .from('synonym_terms')
        .delete()
        .eq('id', id)
      
      if (error) throw error
      return true
    } catch (error) {
      console.error('Error deleting synonym:', error)
      throw error
    }
  }

  // Product Operations
  static async getProducts() {
    try {
      const { data, error } = await supabase
        .from('products')
        .select('*')
        .order('created_at', { ascending: false })
      
      if (error) throw error
      return data
    } catch (error) {
      console.error('Error fetching products:', error)
      throw error
    }
  }

  static async updateProductStatus(id: string, status: string, reviewerId: string) {
    try {
      const { data, error } = await supabase
        .from('products')
        .update({ 
          status, 
          reviewed_by: reviewerId,
          reviewed_at: new Date().toISOString()
        })
        .eq('id', id)
        .select()
      
      if (error) throw error
      return data[0]
    } catch (error) {
      console.error('Error updating product status:', error)
      throw error
    }
  }

  // System Settings Operations
  static async getSystemSettings() {
    try {
      const { data, error } = await supabase
        .from('system_settings')
        .select('*')
        .single()
      
      if (error && error.code !== 'PGRST116') throw error
      return data
    } catch (error) {
      console.error('Error fetching system settings:', error)
      throw error
    }
  }

  static async updateSystemSettings(settings: any) {
    try {
      const { data, error } = await supabase
        .from('system_settings')
        .upsert([settings])
        .select()
      
      if (error) throw error
      return data[0]
    } catch (error) {
      console.error('Error updating system settings:', error)
      throw error
    }
  }

  static async getRegexRules() {
    try {
      const { data, error } = await supabase
        .from('regex_rules')
        .select('*')
        .order('created_at', { ascending: false })
      
      if (error) throw error
      return data
    } catch (error) {
      console.error('Error fetching regex rules:', error)
      throw error
    }
  }

  static async getKeywordRules() {
    try {
      const { data, error } = await supabase
        .from('keyword_rules')
        .select('*')
        .order('created_at', { ascending: false })
      
      if (error) throw error
      return data
    } catch (error) {
      console.error('Error fetching keyword rules:', error)
      throw error
    }
  }
}
