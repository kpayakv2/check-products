/**
 * Single Source of Truth for Types (ADR 0004)
 *
 * Pillar 1: Database types generated from PostgreSQL (127.0.0.1:54325) -> database.types.ts
 * Pillar 2: AI inference types generated from FastAPI (127.0.0.1:8000) -> fastapi.types.ts
 * Centralized Application & View types for Next.js frontend
 */

import type { Database, Tables, TablesInsert, TablesUpdate, Json } from './database.types'
import type { components, operations, paths } from './fastapi.types'

// Re-export core pillars
export type { Database, Tables, TablesInsert, TablesUpdate, Json }
export type { components as FastAPIComponents, operations as FastAPIOperations, paths as FastAPIPaths }

// Re-export import wizard types
export * from './import'

// ==========================================
// Table Row Types (Direct from PostgreSQL)
// ==========================================
export type TaxonomyNodeRow = Tables<'taxonomy_nodes'>
export type ProductRow = Tables<'products'>
export type KeywordRuleRow = Tables<'keyword_rules'>
export type ImportRow = Tables<'imports'>
export type SimilarityMatchRow = Tables<'similarity_matches'>
export type ProductCategorySuggestionRow = Tables<'product_category_suggestions'>
export type AuditLogRow = Tables<'audit_logs'>
export type RegexRuleRow = Tables<'regex_rules'>
export type SynonymLemmaRow = Tables<'synonym_lemmas'>
export type SynonymTermRow = Tables<'synonym_terms'>
export type SystemSettingRow = Tables<'system_settings'>
export type MlTrainingHistoryRow = Tables<'ml_training_history'>
export type HumanFeedbackRow = Tables<'human_feedback'>
export type ReviewHistoryRow = Tables<'review_history'>

// ==========================================
// Domain & Application Types
// ==========================================

export interface TaxonomyNode {
  id: string
  code: string
  name_th: string
  name_en?: string | null
  description?: string | null
  parent_id?: string | null
  level: number | null
  sort_order: number | null
  path?: string | null
  short_code?: string | null
  keywords?: string[] | null
  metadata?: any
  is_active: boolean | null
  created_by?: string | null
  updated_by?: string | null
  created_at: string | null
  updated_at: string | null
  children?: TaxonomyNode[]
}

import type { ProductStatus } from '@/utils/product-status'
export type { ProductStatus }
export { PENDING_REVIEW_STATUSES } from '@/utils/product-status'

export interface ProductAttribute {
  id: string
  product_id: string
  attribute_name: string
  attribute_value: string
  attribute_type: 'text' | 'number' | 'boolean' | 'date' | string
  created_by?: string | null
  created_at: string | null
}

export interface Product {
  id: string
  name_th: string
  name_en?: string | null
  description?: string | null
  category_id?: string | null
  brand?: string | null
  model?: string | null
  sku?: string | null
  price?: number | null
  barcode?: string | null
  unit_name?: string | null
  vat_flag?: string | null
  pos_exported_at?: string | null
  embedding?: string | number[] | null
  keywords?: string[] | null
  metadata?: any
  status: ProductStatus | string | null
  confidence_score?: number | null
  import_batch_id?: string | null
  reviewed_by?: string | null
  reviewed_at?: string | null
  created_by?: string | null
  updated_by?: string | null
  created_at: string | null
  updated_at: string | null
  category?: TaxonomyNode | null
  attributes?: ProductAttribute[]
}

export interface KeywordRule {
  id: string
  code?: string
  name: string
  description?: string | null
  keywords: string[]
  category_id?: string | null
  priority?: number | null
  match_type: 'contains' | 'exact' | 'regex' | 'fuzzy' | 'mined_legacy' | string | null
  confidence_score?: number | null
  is_active?: boolean | null
  created_by?: string | null
  updated_by?: string | null
  created_at?: string | null
  updated_at?: string | null
  category?: TaxonomyNode | null
  taxonomy_nodes?: {
    name_th: string
  } | null
}

export interface SynonymTerm {
  id: string
  lemma_id?: string | null
  term: string
  is_primary?: boolean | null
  confidence_score?: number | null
  source?: 'manual' | 'auto' | 'imported' | 'ml' | string | null
  language?: string | null
  is_verified?: boolean | null
  usage_count?: number | null
  created_by?: string | null
  created_at?: string | null
}

export interface Synonym {
  id: string
  code: string
  name_th: string
  name_en?: string | null
  description?: string | null
  category_id?: string | null
  is_active: boolean | null
  created_by?: string | null
  updated_by?: string | null
  created_at: string | null
  updated_at: string | null
  category?: TaxonomyNode | null
  terms?: SynonymTerm[]
}

export interface SynonymCategoryMap {
  id: string
  lemma_id: string
  category_id: string
  weight: number
  created_by?: string | null
  created_at: string | null
}

export interface ProductCategorySuggestion {
  id: string
  product_id?: string | null
  suggested_category_id: string
  confidence_score?: number | null
  suggestion_method?: 'keyword_rule' | 'ml_model' | 'similarity' | 'manual' | 'recheck_legacy' | string | null
  rule_id?: string | null
  metadata?: any
  is_accepted?: boolean | null
  reviewed_by?: string | null
  reviewed_at?: string | null
  created_at: string | null
  product?: Product | null
  suggested_category?: TaxonomyNode | null
  rule?: KeywordRule | null
}

export interface ImportRecord {
  id: string
  name: string
  description?: string | null
  file_name?: string | null
  file_size?: number | null
  file_type?: string | null
  total_records?: number | null
  processed_records?: number | null
  success_records?: number | null
  error_records?: number | null
  status: 'pending' | 'processing' | 'completed' | 'failed' | string | null
  error_details?: any
  metadata?: any
  created_by?: string | null
  started_at?: string | null
  completed_at?: string | null
  created_at: string | null
}

export type Import = ImportRecord

export interface AuditLog {
  id: string
  table_name: string
  record_id: string
  action: 'INSERT' | 'UPDATE' | 'DELETE' | string
  old_values?: any
  new_values?: any
  changed_fields?: string[] | null
  user_id?: string | null
  user_agent?: string | null
  ip_address?: unknown
  session_id?: string | null
  created_at: string | null
}

export interface SimilarityMatch {
  id: string
  product_a_id: string
  product_b_id: string
  similarity_score: number
  match_type: 'semantic' | 'exact' | 'fuzzy' | 'keyword' | 'rule_numeric_mismatch' | string | null
  algorithm: 'cosine' | 'euclidean' | 'jaccard' | 'legacy_embedding_scan' | string | null
  is_duplicate?: boolean | null
  reviewed: boolean | null
  reviewed_by?: string | null
  reviewed_at?: string | null
  metadata?: any
  created_at: string | null
  product_a?: Product | null
  product_b?: Product | null
}

export interface ReviewHistory {
  id: string
  product_id?: string | null
  reviewer_id?: string | null
  action: 'approved' | 'rejected' | 'modified' | 'category_changed' | string
  old_category_id?: string | null
  new_category_id?: string | null
  comments?: string | null
  metadata?: any
  created_at: string | null
  product?: Product | null
  old_category?: TaxonomyNode | null
  new_category?: TaxonomyNode | null
}

export interface DashboardStats {
  totalCategories: number
  totalSynonyms: number
  pendingDedup: number
  pendingCategory: number
  pendingProducts: number
  approvedProducts: number
  rejectedProducts: number
  duplicatePairs: number
  duplicatePairsReviewed: number
  reviewsToday: number
  recheckAgreement: { total: number; agreed: number } | null
}

// ==========================================
// POS Export Types
// ==========================================

export interface TaxonomyNodeLite {
  id: string
  name_th: string
  parent_id: string | null
  code?: string
  level?: number
}

export interface ExportableProduct {
  name_th: string
  sku: string | null
  barcode: string | null
  unit_name: string | null
  vat_flag: string | null
  price: number | null
  category_id: string | null
  id?: string
  pos_exported_at?: string | null
  updated_at?: string | null
  category?: TaxonomyNodeLite | null
}

export interface PosExportRow {
  itemGroup: string
  subGroup1: string
  subGroup2: string
  itemId: string
  barcode: string
  itemName: string
  unitName: string
  price: number | null
  vat: string
}

export interface ItemIdCandidate {
  id: string
  name_th: string
  sku: string | null
  mainCategory: string | null
  barcode?: string | null
}

// ==========================================
// General API Response Type
// ==========================================

export interface ApiResponse<T = any> {
  success: boolean
  data?: T
  error?: string
  message?: string
  total?: number
  page?: number
  pageSize?: number
}

// ==========================================
// AI Service Contracts (FastAPI)
// ==========================================

export type CategoryRequest = components['schemas']['CategoryRequest']
export type CategoryResponse = components['schemas']['CategoryResponse']
export type BatchCategoryRequest = components['schemas']['BatchCategoryRequest']
export type BatchCategoryResponse = components['schemas']['BatchCategoryResponse']
export type CleanTextRequest = components['schemas']['CleanTextRequest']
export type CleanTextResponse = components['schemas']['CleanTextResponse']
export type CleanTextResult = components['schemas']['CleanTextResult']
export type ProductMatchResult = components['schemas']['ProductMatchResult']
export type ImportDeduplicationRequest = components['schemas']['ImportDeduplicationRequest']
export type FileDeduplicationRequest = components['schemas']['FileDeduplicationRequest']
export type FileDeduplicationResponse = components['schemas']['FileDeduplicationResponse']
export type ScanInternalRequest = components['schemas']['ScanInternalRequest']
export type ScanStartResponse = components['schemas']['ScanStartResponse']
export type ScanTaskStatusResponse = components['schemas']['ScanTaskStatusResponse']
