import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

export const supabase = createClient(supabaseUrl, supabaseAnonKey)

// Types สำหรับ Database Schema
export interface TaxonomyNode {
  id: string
  code: string
  name_th: string
  name_en?: string
  description?: string
  parent_id?: string
  level: number
  sort_order: number
  path?: string
  keywords?: string[]
  metadata?: any
  is_active: boolean
  created_by?: string
  updated_by?: string
  created_at: string
  updated_at: string
  children?: TaxonomyNode[]
}

export interface Synonym {
  id: string
  code: string
  name_th: string
  name_en?: string
  description?: string
  category_id?: string
  is_active: boolean
  created_by?: string
  updated_by?: string
  created_at: string
  updated_at: string
  category?: TaxonomyNode
  terms?: SynonymTerm[]
}

export interface SynonymTerm {
  id: string
  lemma_id: string
  term: string
  is_primary: boolean
  confidence_score: number
  source: 'manual' | 'auto' | 'imported' | 'ml'
  language: string
  is_verified: boolean
  created_by?: string
  created_at: string
}

export interface SynonymCategoryMap {
  id: string
  lemma_id: string
  category_id: string
  weight: number
  created_by?: string
  created_at: string
}

export interface KeywordRule {
  id: string
  name: string
  description?: string
  keywords: string[]
  category_id: string
  priority: number
  match_type: 'contains' | 'exact' | 'regex' | 'fuzzy'
  confidence_score: number
  is_active: boolean
  created_by?: string
  updated_by?: string
  created_at: string
  updated_at: string
  category?: TaxonomyNode
}

export interface Product {
  id: string
  name_th: string
  name_en?: string
  description?: string
  category_id?: string
  brand?: string
  model?: string
  sku?: string
  price?: number
  embedding?: number[]
  keywords?: string[]
  metadata?: any
  status: 'pending' | 'approved' | 'rejected' | 'draft'
  confidence_score?: number
  import_batch_id?: string
  reviewed_by?: string
  reviewed_at?: string
  created_by?: string
  updated_by?: string
  created_at: string
  updated_at: string
  category?: TaxonomyNode
  attributes?: ProductAttribute[]
}

export interface ProductAttribute {
  id: string
  product_id: string
  attribute_name: string
  attribute_value: string
  attribute_type: 'text' | 'number' | 'boolean' | 'date'
  created_by?: string
  created_at: string
}

export interface ProductCategorySuggestion {
  id: string
  product_id: string
  suggested_category_id: string
  confidence_score: number
  suggestion_method: 'keyword_rule' | 'ml_model' | 'similarity' | 'manual'
  rule_id?: string
  metadata?: any
  is_accepted?: boolean
  reviewed_by?: string
  reviewed_at?: string
  created_at: string
  product?: Product
  suggested_category?: TaxonomyNode
  rule?: KeywordRule
}

export interface Import {
  id: string
  name: string
  description?: string
  file_name?: string
  file_size?: number
  file_type?: 'csv' | 'xlsx' | 'json'
  total_records: number
  processed_records: number
  success_records: number
  error_records: number
  status: 'pending' | 'processing' | 'completed' | 'failed'
  error_details?: any
  metadata?: any
  created_by?: string
  started_at?: string
  completed_at?: string
  created_at: string
}

export interface AuditLog {
  id: string
  table_name: string
  record_id: string
  action: 'INSERT' | 'UPDATE' | 'DELETE'
  old_values?: any
  new_values?: any
  changed_fields?: string[]
  user_id?: string
  user_agent?: string
  ip_address?: string
  session_id?: string
  created_at: string
}

export interface SimilarityMatch {
  id: string
  product_a_id: string
  product_b_id: string
  similarity_score: number
  match_type: 'semantic' | 'exact' | 'fuzzy' | 'keyword'
  algorithm: 'cosine' | 'euclidean' | 'jaccard'
  is_duplicate: boolean
  reviewed: boolean
  reviewed_by?: string
  reviewed_at?: string
  metadata?: any
  created_at: string
  product_a?: Product
  product_b?: Product
}

export interface ReviewHistory {
  id: string
  product_id: string
  reviewer_id?: string
  action: 'approved' | 'rejected' | 'modified' | 'category_changed'
  old_category_id?: string
  new_category_id?: string
  comments?: string
  metadata?: any
  created_at: string
  product?: Product
  old_category?: TaxonomyNode
  new_category?: TaxonomyNode
}

// Database Operations
export class DatabaseService {
  // Taxonomy Nodes
  static async getTaxonomyTree(options?: {
    include_inactive?: boolean
    limit?: number
    offset?: number
  }): Promise<TaxonomyNode[]> {
    let query = supabase
      .from('taxonomy_nodes')
      .select('*')
      .order('level', { ascending: true })
      .order('sort_order', { ascending: true })

    if (!options?.include_inactive) {
      query = query.eq('is_active', true)
    }

    if (options?.limit) {
      query = query.limit(options.limit)
    }

    if (options?.offset) {
      query = query.range(options.offset, options.offset + (options.limit || 100) - 1)
    }

    const { data, error } = await query
    if (error) throw error
    return this.buildTaxonomyTree(data || [])
  }

  static async getTaxonomyNodes(options: {
    parent_id?: string
    level?: number
    include_inactive?: boolean
    limit?: number
    offset?: number
  }): Promise<TaxonomyNode[]> {
    let query = supabase
      .from('taxonomy_nodes')
      .select('*')
      .order('sort_order', { ascending: true })

    if (options.parent_id) {
      query = query.eq('parent_id', options.parent_id)
    }

    if (options.level !== undefined) {
      query = query.eq('level', options.level)
    }

    if (!options.include_inactive) {
      query = query.eq('is_active', true)
    }

    if (options.limit) {
      query = query.limit(options.limit)
    }

    if (options.offset) {
      query = query.range(options.offset, options.offset + (options.limit || 100) - 1)
    }

    const { data, error } = await query
    if (error) throw error
    return data || []
  }

  static async getTaxonomyNode(id: string): Promise<TaxonomyNode | null> {
    const { data, error } = await supabase
      .from('taxonomy_nodes')
      .select('*')
      .eq('id', id)
      .single()

    if (error) {
      if (error.code === 'PGRST116') return null // Not found
      throw error
    }
    return data
  }

  static async getNextSortOrder(parent_id?: string): Promise<number> {
    let query = supabase
      .from('taxonomy_nodes')
      .select('sort_order')
      .order('sort_order', { ascending: false })
      .limit(1)

    if (parent_id) {
      query = query.eq('parent_id', parent_id)
    } else {
      query = query.is('parent_id', null)
    }

    const { data, error } = await query
    if (error) throw error
    
    return data && data.length > 0 ? data[0].sort_order + 1 : 0
  }

  static async createTaxonomyNode(node: Partial<TaxonomyNode>): Promise<TaxonomyNode> {
    const nodeData = {
      ...node,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    }

    const { data, error } = await supabase
      .from('taxonomy_nodes')
      .insert(nodeData)
      .select()
      .single()

    if (error) throw error
    return data
  }

  static async updateTaxonomyNode(id: string, updates: Partial<TaxonomyNode>): Promise<TaxonomyNode> {
    const { data, error } = await supabase
      .from('taxonomy_nodes')
      .update(updates)
      .eq('id', id)
      .select()
      .single()

    if (error) throw error
    return data
  }

  static async deleteTaxonomyNode(id: string): Promise<void> {
    const { error } = await supabase
      .from('taxonomy_nodes')
      .delete()
      .eq('id', id)

    if (error) throw error
  }

  // Synonyms
  static async getSynonyms(categoryId?: string): Promise<Synonym[]> {
    let query = supabase
      .from('synonym_lemmas')
      .select(`
        *,
        category:taxonomy_nodes(*),
        terms:synonym_terms(*)
      `)
      .order('name_th', { ascending: true })

    if (categoryId) {
      query = query.eq('category_id', categoryId)
    }

    const { data, error } = await query
    if (error) {
      console.error('Supabase getSynonyms error:', error)
      throw error
    }
    return data || []
  }

  static async createSynonym(synonym: Partial<Synonym>): Promise<Synonym> {
    const { data, error } = await supabase
      .from('synonym_lemmas')
      .insert(synonym)
      .select(`
        *,
        category:taxonomy_nodes(*),
        terms:synonym_terms(*)
      `)
      .single()

    if (error) {
      console.error('Supabase createSynonym error:', error)
      throw error
    }
    return data
  }

  static async updateSynonym(id: string, updates: Partial<Synonym>): Promise<Synonym> {
    const { data, error } = await supabase
      .from('synonym_lemmas')
      .update(updates)
      .eq('id', id)
      .select(`
        *,
        category:taxonomy_nodes(*),
        terms:synonym_terms(*)
      `)
      .single()

    if (error) {
      console.error('Supabase updateSynonym error:', error)
      throw error
    }
    return data
  }

  static async deleteSynonym(id: string): Promise<void> {
    const { error } = await supabase
      .from('synonym_lemmas')
      .delete()
      .eq('id', id)

    if (error) {
      console.error('Supabase deleteSynonym error:', error)
      throw error
    }
  }

  // Synonym Terms
  static async createSynonymTerm(termData: Partial<SynonymTerm>): Promise<SynonymTerm> {
    const { data, error } = await supabase
      .from('synonym_terms')
      .insert({
        ...termData,
        created_at: new Date().toISOString()
      })
      .select()
      .single()

    if (error) {
      console.error('Supabase createSynonymTerm error:', error)
      throw error
    }
    return data
  }

  static async getSynonymTerms(synonymId: string): Promise<SynonymTerm[]> {
    const { data, error } = await supabase
      .from('synonym_terms')
      .select('*')
      .eq('lemma_id', synonymId)
      .order('is_primary', { ascending: false })
      .order('created_at', { ascending: true })

    if (error) throw error
    return data || []
  }

  // Products
  static async getProducts(status?: Product['status'], limit = 50): Promise<Product[]> {
    let query = supabase
      .from('products')
      .select(`
        *,
        category:taxonomy_nodes(*),
        attributes:product_attributes(*)
      `)
      .order('created_at', { ascending: false })
      .limit(limit)

    if (status) {
      query = query.eq('status', status)
    }

    const { data, error } = await query
    if (error) throw error
    return data || []
  }

  static async updateProductStatus(id: string, status: Product['status'], reviewerId?: string): Promise<Product> {
    const updates: any = { 
      status,
      reviewed_at: new Date().toISOString()
    }
    
    if (reviewerId) {
      updates.reviewed_by = reviewerId
    }

    const { data, error } = await supabase
      .from('products')
      .update(updates)
      .eq('id', id)
      .select(`
        *,
        category:taxonomy_nodes(*),
        attributes:product_attributes(*)
      `)
      .single()

    if (error) throw error
    return data
  }

  // Similarity Matches
  static async getSimilarityMatches(productId?: string): Promise<SimilarityMatch[]> {
    let query = supabase
      .from('similarity_matches')
      .select(`
        *,
        product_a:products!product_a_id(*),
        product_b:products!product_b_id(*)
      `)
      .order('similarity_score', { ascending: false })

    if (productId) {
      query = query.or(`product_a_id.eq.${productId},product_b_id.eq.${productId}`)
    }

    const { data, error } = await query
    if (error) throw error
    return data || []
  }

  static async createSimilarityMatch(matchData: Partial<SimilarityMatch>): Promise<SimilarityMatch> {
    const { data, error } = await supabase
      .from('similarity_matches')
      .insert({
        ...matchData,
        created_at: new Date().toISOString()
      })
      .select()
      .single()

    if (error) throw error
    return data
  }

  // Import Management
  static async createImport(importData: Partial<Import>): Promise<Import> {
    const { data, error } = await supabase
      .from('imports')
      .insert(importData)
      .select()
      .single()

    if (error) throw error
    return data
  }

  static async updateImport(id: string, updates: Partial<Import>): Promise<Import> {
    const { data, error } = await supabase
      .from('imports')
      .update(updates)
      .eq('id', id)
      .select()
      .single()

    if (error) throw error
    return data
  }

  static async createProduct(productData: Partial<Product>): Promise<Product> {
    const { data, error } = await supabase
      .from('products')
      .insert(productData)
      .select()
      .single()

    if (error) throw error
    return data
  }

  static async createProductCategorySuggestion(suggestionData: Partial<ProductCategorySuggestion>): Promise<ProductCategorySuggestion> {
    const { data, error } = await supabase
      .from('product_category_suggestions')
      .insert(suggestionData)
      .select()
      .single()

    if (error) throw error
    return data
  }

  static async createProductAttribute(attributeData: Partial<ProductAttribute>): Promise<ProductAttribute> {
    const { data, error } = await supabase
      .from('product_attributes')
      .insert(attributeData)
      .select()
      .single()

    if (error) throw error
    return data
  }

  // Settings and Rules Management
  static async getRegexRules(): Promise<any[]> {
    const { data, error } = await supabase
      .from('regex_rules')
      .select('*')
      .order('priority', { ascending: false })

    if (error) throw error
    return data || []
  }

  static async getKeywordRules(): Promise<any[]> {
    const { data, error } = await supabase
      .from('keyword_rules')
      .select(`
        *,
        category:taxonomy_nodes(id, name_th, name_en)
      `)
      .order('priority', { ascending: false })

    if (error) throw error
    return data || []
  }

  static async getSystemSettings(): Promise<any> {
    const { data, error } = await supabase
      .from('system_settings')
      .select('*')
      .single()

    if (error && error.code !== 'PGRST116') throw error
    return data
  }

  static async updateSystemSettings(settings: any): Promise<any> {
    const { data, error } = await supabase
      .from('system_settings')
      .upsert(settings)
      .select()
      .single()

    if (error) throw error
    return data
  }

  static async createRegexRule(rule: any): Promise<any> {
    const { data, error } = await supabase
      .from('regex_rules')
      .insert(rule)
      .select()
      .single()

    if (error) throw error
    return data
  }

  static async updateRegexRule(id: string, updates: any): Promise<any> {
    const { data, error } = await supabase
      .from('regex_rules')
      .update(updates)
      .eq('id', id)
      .select()
      .single()

    if (error) throw error
    return data
  }

  static async deleteRegexRule(id: string): Promise<void> {
    const { error } = await supabase
      .from('regex_rules')
      .delete()
      .eq('id', id)

    if (error) throw error
  }

  static async createKeywordRule(rule: any): Promise<any> {
    const { data, error } = await supabase
      .from('keyword_rules')
      .insert(rule)
      .select()
      .single()

    if (error) throw error
    return data
  }

  static async updateKeywordRule(id: string, updates: any): Promise<any> {
    const { data, error } = await supabase
      .from('keyword_rules')
      .update(updates)
      .eq('id', id)
      .select()
      .single()

    if (error) throw error
    return data
  }

  static async deleteKeywordRule(id: string): Promise<void> {
    const { error } = await supabase
      .from('keyword_rules')
      .delete()
      .eq('id', id)

    if (error) throw error
  }

  // Review History
  static async createReviewHistory(historyData: Partial<ReviewHistory>): Promise<ReviewHistory> {
    const { data, error } = await supabase
      .from('review_history')
      .insert({
        ...historyData,
        created_at: new Date().toISOString()
      })
      .select()
      .single()

    if (error) throw error
    return data
  }

  // Helper method to build taxonomy tree
  private static buildTaxonomyTree(nodes: TaxonomyNode[]): TaxonomyNode[] {
    const nodeMap = new Map<string, TaxonomyNode>()
    const rootNodes: TaxonomyNode[] = []

    // สร้าง map และเพิ่ม children array
    nodes.forEach(node => {
      nodeMap.set(node.id, { ...node, children: [] })
    })

    // สร้าง tree structure
    nodes.forEach(node => {
      const nodeWithChildren = nodeMap.get(node.id)!
      
      if (node.parent_id) {
        const parent = nodeMap.get(node.parent_id)
        if (parent) {
          parent.children!.push(nodeWithChildren)
        }
      } else {
        rootNodes.push(nodeWithChildren)
      }
    })

    return rootNodes
  }
}
