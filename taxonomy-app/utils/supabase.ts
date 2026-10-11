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

import type { Database } from '@/types/database.types'
import type {
  TaxonomyNode,
  Synonym,
  SynonymTerm,
  SynonymCategoryMap,
  KeywordRule,
  Product,
  ProductAttribute,
  ProductCategorySuggestion,
  Import,
  AuditLog,
  SimilarityMatch,
  ReviewHistory,
  DashboardStats,
} from '@/types'

export type {
  TaxonomyNode,
  Synonym,
  SynonymTerm,
  SynonymCategoryMap,
  KeywordRule,
  Product,
  ProductAttribute,
  ProductCategorySuggestion,
  Import,
  AuditLog,
  SimilarityMatch,
  ReviewHistory,
  DashboardStats,
}

export const supabase = createClient<Database>(getSupabaseUrl(), supabaseAnonKey)


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
    return this.buildTaxonomyTree((data || []) as unknown as TaxonomyNode[])
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
    return (data || []) as unknown as Synonym[]
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
    return { products: (data || []) as unknown as Product[], total: count || 0 }
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
          parent.children = parent.children || []
          parent.children.push(nodeWithChildren)
        }
      } else {
        rootNodes.push(nodeWithChildren)
      }
    })

    return rootNodes
  }
}

