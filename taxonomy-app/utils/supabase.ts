import { createClient } from '@supabase/supabase-js'
import { ProductStatus, PENDING_REVIEW_STATUSES } from './product-status'

export type { ProductStatus }
export { PENDING_REVIEW_STATUSES }

const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

// สร้าง Supabase URL แบบ dynamic เพื่อรองรับ LAN access
// - บนเครื่อง server: http://localhost:3000/api/supabase
// - บนเครื่อง LAN:   http://192.168.1.80:3000/api/supabase
// Next.js proxy (next.config.js) จะส่งต่อ /api/supabase/* → http://127.0.0.1:54331/*
const getSupabaseUrl = (): string => {
  // Server-side rendering: ใช้ env variable โดยตรง
  if (typeof window === 'undefined') {
    return process.env.SUPABASE_URL || 'http://127.0.0.1:54331'
  }
  // Client-side: ใช้ origin ของ browser + proxy path (รองรับ LAN อัตโนมัติ)
  return `${window.location.origin}/api/supabase`
}

export const supabase = createClient(getSupabaseUrl(), supabaseAnonKey)

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
  status: ProductStatus
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

/** ตัวเลขบนหน้าแรก — ทุกค่าต้องมาจากฐานข้อมูล ไม่มีค่าที่ฝังไว้ในโค้ด */
export interface DashboardStats {
  totalCategories: number
  totalSynonyms: number
  /** รอตรวจของซ้ำ (ด่าน 1) */
  pendingDedup: number
  /** รอตรวจหมวดหมู่ (ด่าน 2) */
  pendingCategory: number
  /** ผลรวมของสองด่าน */
  pendingProducts: number
  approvedProducts: number
  rejectedProducts: number
  duplicatePairs: number
  duplicatePairsReviewed: number
  /** จำนวนการตรวจที่บันทึกลง review_history วันนี้ */
  reviewsToday: number
  /** null = ยังไม่เคยรันตรวจซ้ำ จึงยังไม่มีตัวเลขให้แสดง */
  recheckAgreement: { total: number; agreed: number } | null
}

/**
 * ตัวกรอง `or(...)` ของ PostgREST คั่นเงื่อนไขด้วยจุลภาคและใช้วงเล็บจัดกลุ่ม
 * ถ้าปล่อยอักขระพวกนี้ติดไปกับคำค้น ตัวกรองจะเพี้ยนหรือ error ทั้งชุด
 */
const sanitizeSearchTerm = (term?: string): string =>
  (term || '').replace(/[,()*\\%]/g, ' ').trim()

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

  static async searchTaxonomyNodes(query: string, limit = 10): Promise<TaxonomyNode[]> {
    const { data, error } = await supabase
      .from('taxonomy_nodes')
      .select('*')
      .or(`name_th.ilike.%${query}%,code.ilike.%${query}%`)
      .eq('is_active', true)
      .limit(limit)

    if (error) throw error
    return data || []
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
  /**
   * ค้นและแบ่งหน้าที่ฝั่งฐานข้อมูล — สตอกมี 3,000+ แถว การดึงมากรองในเบราว์เซอร์
   * (แบบที่ `getProducts` ทำ) จะเห็นแค่หน้าแรกเท่านั้นและค้นไม่เจอของที่เหลือ
   */
  static async searchProducts(options: {
    status?: ProductStatus | ProductStatus[]
    categoryId?: string
    search?: string
    limit?: number
    offset?: number
  } = {}): Promise<{ products: Product[]; total: number }> {
    const limit = options.limit ?? 50
    const offset = options.offset ?? 0

    let query = supabase
      .from('products')
      .select('*, category:taxonomy_nodes(*)', { count: 'exact' })
      .order('created_at', { ascending: false })
      .range(offset, offset + limit - 1)

    if (options.status) {
      query = Array.isArray(options.status)
        ? query.in('status', options.status)
        : query.eq('status', options.status)
    }

    if (options.categoryId) {
      query = query.eq('category_id', options.categoryId)
    }

    const term = sanitizeSearchTerm(options.search)
    if (term) {
      query = query.or(
        `name_th.ilike.%${term}%,name_en.ilike.%${term}%,brand.ilike.%${term}%,sku.ilike.%${term}%`
      )
    }

    const { data, count, error } = await query
    if (error) throw error
    return { products: data || [], total: count || 0 }
  }

  /** นับแยกรายสถานะด้วย head query — ไม่ดึงแถวจริงมาเลย */
  static async getProductStatusCounts(): Promise<Record<ProductStatus, number>> {
    const statuses: ProductStatus[] = [
      'pending_review_dedup',
      'pending_review_category',
      'approved',
      'rejected',
      'draft'
    ]

    const counts = await Promise.all(
      statuses.map(async status => {
        const { count } = await supabase
          .from('products')
          .select('*', { count: 'exact', head: true })
          .eq('status', status)
        return [status, count || 0] as const
      })
    )

    return Object.fromEntries(counts) as Record<ProductStatus, number>
  }

  static async getProducts(status?: ProductStatus | ProductStatus[], limit = 50): Promise<Product[]> {
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
      query = Array.isArray(status) ? query.in('status', status) : query.eq('status', status)
    }

    const { data, error } = await query
    if (error) throw error
    return data || []
  }

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

  // Dashboard Statistics
  static async getDashboardStats(): Promise<DashboardStats> {
    const today = new Date()
    today.setHours(0, 0, 0, 0)
    const todayIso = today.toISOString()

    const [
      { count: catCount },
      { count: synCount },
      { count: pendingDedupCount },
      { count: pendingCategoryCount },
      { count: approvedCount },
      { count: rejectedCount },
      { count: duplicatePairCount },
      { count: duplicatePairReviewedCount },
      { count: reviewedTodayCount },
      agreement
    ] = await Promise.all([
      supabase.from('taxonomy_nodes').select('*', { count: 'exact', head: true }),
      supabase.from('synonym_lemmas').select('*', { count: 'exact', head: true }),
      supabase.from('products').select('*', { count: 'exact', head: true }).eq('status', 'pending_review_dedup'),
      supabase.from('products').select('*', { count: 'exact', head: true }).eq('status', 'pending_review_category'),
      supabase.from('products').select('*', { count: 'exact', head: true }).eq('status', 'approved'),
      supabase.from('products').select('*', { count: 'exact', head: true }).eq('status', 'rejected'),
      supabase.from('similarity_matches').select('*', { count: 'exact', head: true }),
      supabase.from('similarity_matches').select('*', { count: 'exact', head: true }).eq('reviewed', true),
      supabase.from('review_history').select('*', { count: 'exact', head: true }).gte('created_at', todayIso),
      this.getRecheckAgreement()
    ])

    const pendingDedup = pendingDedupCount || 0
    const pendingCategory = pendingCategoryCount || 0

    return {
      totalCategories: catCount || 0,
      totalSynonyms: synCount || 0,
      pendingDedup,
      pendingCategory,
      pendingProducts: pendingDedup + pendingCategory,
      approvedProducts: approvedCount || 0,
      rejectedProducts: rejectedCount || 0,
      duplicatePairs: duplicatePairCount || 0,
      duplicatePairsReviewed: duplicatePairReviewedCount || 0,
      reviewsToday: reviewedTodayCount || 0,
      recheckAgreement: agreement
    }
  }

  /**
   * สัดส่วนที่ AI ตรวจซ้ำแล้วเห็นตรงกับหมวดที่คนจัดไว้ — คำนวณในฐานข้อมูล
   * (`recheck_agreement_stats()`) เพราะต้องเทียบสินค้า 3,000+ แถว
   * คืน null เมื่อยังไม่มีข้อมูลตรวจซ้ำ เพื่อให้หน้าเว็บเลือกที่จะไม่แสดงตัวเลขได้
   */
  static async getRecheckAgreement(): Promise<{ total: number; agreed: number } | null> {
    const { data, error } = await supabase.rpc('recheck_agreement_stats')

    if (error) {
      console.error('recheck_agreement_stats failed:', error)
      return null
    }

    const row = Array.isArray(data) ? data[0] : data
    if (!row || !row.total) return null
    return { total: Number(row.total), agreed: Number(row.agreed) }
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
