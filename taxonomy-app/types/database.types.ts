
export type Json = string | number | boolean | null | { [key: string]: Json | undefined } | Json[]

export type Database = {
  
  "graphql_public": {
          Tables: {
            [_ in never]: never
          }
          Views: {
            [_ in never]: never
          }
          Functions: {
            "graphql":
{ Args: { "extensions"?: Json,"operationName"?: string,"query"?: string,"variables"?: Json }; Returns: Json
                           }
          }
          Enums: {
            [_ in never]: never
          }
          CompositeTypes: {
            [_ in never]: never
          }
        },"public": {
          Tables: {
            "audit_logs": {
                  Row: {
                    "action": string,"changed_fields": (string)[] | null,"created_at": string | null,"id": string,"ip_address": unknown,"new_values": Json | null,"old_values": Json | null,"record_id": string,"session_id": string | null,"table_name": string,"user_agent": string | null,"user_id": string | null
                  }
                  ComputedFields: never
                  Insert: {
                    "action": string,"changed_fields"?: (string)[] | null,"created_at"?: string | null,"id"?: string,"ip_address"?: unknown,"new_values"?: Json | null,"old_values"?: Json | null,"record_id": string,"session_id"?: string | null,"table_name": string,"user_agent"?: string | null,"user_id"?: string | null
                  }
                  Update: {
                    "action"?: string,"changed_fields"?: (string)[] | null,"created_at"?: string | null,"id"?: string,"ip_address"?: unknown,"new_values"?: Json | null,"old_values"?: Json | null,"record_id"?: string,"session_id"?: string | null,"table_name"?: string,"user_agent"?: string | null,"user_id"?: string | null
                  }
                  Relationships: [
                    
                  ]
                },"human_feedback": {
                  Row: {
                    "comments": string | null,"confidence_score": number | null,"created_at": string | null,"human_decision": string,"id": string,"ml_prediction": string | null,"new_product": string,"old_product": string,"processing_time": number | null,"reviewer_id": string | null,"similarity_score": number,"updated_at": string | null
                  }
                  ComputedFields: never
                  Insert: {
                    "comments"?: string | null,"confidence_score"?: number | null,"created_at"?: string | null,"human_decision": string,"id"?: string,"ml_prediction"?: string | null,"new_product": string,"old_product": string,"processing_time"?: number | null,"reviewer_id"?: string | null,"similarity_score": number,"updated_at"?: string | null
                  }
                  Update: {
                    "comments"?: string | null,"confidence_score"?: number | null,"created_at"?: string | null,"human_decision"?: string,"id"?: string,"ml_prediction"?: string | null,"new_product"?: string,"old_product"?: string,"processing_time"?: number | null,"reviewer_id"?: string | null,"similarity_score"?: number,"updated_at"?: string | null
                  }
                  Relationships: [
                    
                  ]
                },"imports": {
                  Row: {
                    "completed_at": string | null,"created_at": string | null,"created_by": string | null,"description": string | null,"error_details": Json | null,"error_records": number | null,"file_name": string | null,"file_size": number | null,"file_type": string | null,"id": string,"metadata": Json | null,"name": string,"processed_records": number | null,"started_at": string | null,"status": string | null,"success_records": number | null,"total_records": number | null
                  }
                  ComputedFields: never
                  Insert: {
                    "completed_at"?: string | null,"created_at"?: string | null,"created_by"?: string | null,"description"?: string | null,"error_details"?: Json | null,"error_records"?: number | null,"file_name"?: string | null,"file_size"?: number | null,"file_type"?: string | null,"id"?: string,"metadata"?: Json | null,"name": string,"processed_records"?: number | null,"started_at"?: string | null,"status"?: string | null,"success_records"?: number | null,"total_records"?: number | null
                  }
                  Update: {
                    "completed_at"?: string | null,"created_at"?: string | null,"created_by"?: string | null,"description"?: string | null,"error_details"?: Json | null,"error_records"?: number | null,"file_name"?: string | null,"file_size"?: number | null,"file_type"?: string | null,"id"?: string,"metadata"?: Json | null,"name"?: string,"processed_records"?: number | null,"started_at"?: string | null,"status"?: string | null,"success_records"?: number | null,"total_records"?: number | null
                  }
                  Relationships: [
                    
                  ]
                },"keyword_rules": {
                  Row: {
                    "category_id": string | null,"code": string,"confidence_score": number | null,"created_at": string | null,"created_by": string | null,"description": string | null,"id": string,"is_active": boolean | null,"keywords": (string)[],"match_type": string | null,"name": string,"priority": number | null,"updated_at": string | null,"updated_by": string | null
                  }
                  ComputedFields: never
                  Insert: {
                    "category_id"?: string | null,"code": string,"confidence_score"?: number | null,"created_at"?: string | null,"created_by"?: string | null,"description"?: string | null,"id"?: string,"is_active"?: boolean | null,"keywords": (string)[],"match_type"?: string | null,"name": string,"priority"?: number | null,"updated_at"?: string | null,"updated_by"?: string | null
                  }
                  Update: {
                    "category_id"?: string | null,"code"?: string,"confidence_score"?: number | null,"created_at"?: string | null,"created_by"?: string | null,"description"?: string | null,"id"?: string,"is_active"?: boolean | null,"keywords"?: (string)[],"match_type"?: string | null,"name"?: string,"priority"?: number | null,"updated_at"?: string | null,"updated_by"?: string | null
                  }
                  Relationships: [
                    {
      foreignKeyName: "keyword_rules_category_id_fkey"
      columns: ["category_id"]
isOneToOne: false
      referencedRelation: "taxonomy_nodes"
      referencedColumns: ["id"]
    }
                  ]
                },"ml_training_history": {
                  Row: {
                    "classes": Json | null,"classification_report": Json | null,"created_at": string | null,"cv_mean_accuracy": number | null,"cv_std_accuracy": number | null,"feature_importance": Json | null,"id": string,"model_type": string,"test_accuracy": number,"test_samples": number,"total_samples": number,"train_accuracy": number,"train_samples": number,"training_date": string
                  }
                  ComputedFields: never
                  Insert: {
                    "classes"?: Json | null,"classification_report"?: Json | null,"created_at"?: string | null,"cv_mean_accuracy"?: number | null,"cv_std_accuracy"?: number | null,"feature_importance"?: Json | null,"id"?: string,"model_type"?: string,"test_accuracy"?: number,"test_samples"?: number,"total_samples"?: number,"train_accuracy"?: number,"train_samples"?: number,"training_date"?: string
                  }
                  Update: {
                    "classes"?: Json | null,"classification_report"?: Json | null,"created_at"?: string | null,"cv_mean_accuracy"?: number | null,"cv_std_accuracy"?: number | null,"feature_importance"?: Json | null,"id"?: string,"model_type"?: string,"test_accuracy"?: number,"test_samples"?: number,"total_samples"?: number,"train_accuracy"?: number,"train_samples"?: number,"training_date"?: string
                  }
                  Relationships: [
                    
                  ]
                },"product_attributes": {
                  Row: {
                    "attribute_name": string,"attribute_type": string | null,"attribute_value": string,"created_at": string | null,"created_by": string | null,"id": string,"product_id": string | null
                  }
                  ComputedFields: never
                  Insert: {
                    "attribute_name": string,"attribute_type"?: string | null,"attribute_value": string,"created_at"?: string | null,"created_by"?: string | null,"id"?: string,"product_id"?: string | null
                  }
                  Update: {
                    "attribute_name"?: string,"attribute_type"?: string | null,"attribute_value"?: string,"created_at"?: string | null,"created_by"?: string | null,"id"?: string,"product_id"?: string | null
                  }
                  Relationships: [
                    {
      foreignKeyName: "product_attributes_product_id_fkey"
      columns: ["product_id"]
isOneToOne: false
      referencedRelation: "products"
      referencedColumns: ["id"]
    }
                  ]
                },"product_category_suggestions": {
                  Row: {
                    "confidence_score": number,"created_at": string | null,"id": string,"is_accepted": boolean | null,"metadata": Json | null,"product_id": string | null,"reviewed_at": string | null,"reviewed_by": string | null,"rule_id": string | null,"suggested_category_id": string | null,"suggestion_method": string
                  }
                  ComputedFields: never
                  Insert: {
                    "confidence_score": number,"created_at"?: string | null,"id"?: string,"is_accepted"?: boolean | null,"metadata"?: Json | null,"product_id"?: string | null,"reviewed_at"?: string | null,"reviewed_by"?: string | null,"rule_id"?: string | null,"suggested_category_id"?: string | null,"suggestion_method": string
                  }
                  Update: {
                    "confidence_score"?: number,"created_at"?: string | null,"id"?: string,"is_accepted"?: boolean | null,"metadata"?: Json | null,"product_id"?: string | null,"reviewed_at"?: string | null,"reviewed_by"?: string | null,"rule_id"?: string | null,"suggested_category_id"?: string | null,"suggestion_method"?: string
                  }
                  Relationships: [
                    {
      foreignKeyName: "product_category_suggestions_product_id_fkey"
      columns: ["product_id"]
isOneToOne: false
      referencedRelation: "products"
      referencedColumns: ["id"]
    },{
      foreignKeyName: "product_category_suggestions_rule_id_fkey"
      columns: ["rule_id"]
isOneToOne: false
      referencedRelation: "keyword_rules"
      referencedColumns: ["id"]
    },{
      foreignKeyName: "product_category_suggestions_suggested_category_id_fkey"
      columns: ["suggested_category_id"]
isOneToOne: false
      referencedRelation: "taxonomy_nodes"
      referencedColumns: ["id"]
    }
                  ]
                },"products": {
                  Row: {
                    "barcode": string | null,"brand": string | null,"category_id": string | null,"confidence_score": number | null,"created_at": string | null,"created_by": string | null,"description": string | null,"embedding": string | null,"id": string,"import_batch_id": string | null,"keywords": (string)[] | null,"metadata": Json | null,"model": string | null,"name_en": string | null,"name_th": string,"pos_exported_at": string | null,"price": number | null,"reviewed_at": string | null,"reviewed_by": string | null,"sku": string | null,"status": string | null,"unit_name": string | null,"updated_at": string | null,"updated_by": string | null,"vat_flag": string
                  }
                  ComputedFields: never
                  Insert: {
                    "barcode"?: string | null,"brand"?: string | null,"category_id"?: string | null,"confidence_score"?: number | null,"created_at"?: string | null,"created_by"?: string | null,"description"?: string | null,"embedding"?: string | null,"id"?: string,"import_batch_id"?: string | null,"keywords"?: (string)[] | null,"metadata"?: Json | null,"model"?: string | null,"name_en"?: string | null,"name_th": string,"pos_exported_at"?: string | null,"price"?: number | null,"reviewed_at"?: string | null,"reviewed_by"?: string | null,"sku"?: string | null,"status"?: string | null,"unit_name"?: string | null,"updated_at"?: string | null,"updated_by"?: string | null,"vat_flag"?: string
                  }
                  Update: {
                    "barcode"?: string | null,"brand"?: string | null,"category_id"?: string | null,"confidence_score"?: number | null,"created_at"?: string | null,"created_by"?: string | null,"description"?: string | null,"embedding"?: string | null,"id"?: string,"import_batch_id"?: string | null,"keywords"?: (string)[] | null,"metadata"?: Json | null,"model"?: string | null,"name_en"?: string | null,"name_th"?: string,"pos_exported_at"?: string | null,"price"?: number | null,"reviewed_at"?: string | null,"reviewed_by"?: string | null,"sku"?: string | null,"status"?: string | null,"unit_name"?: string | null,"updated_at"?: string | null,"updated_by"?: string | null,"vat_flag"?: string
                  }
                  Relationships: [
                    {
      foreignKeyName: "products_category_id_fkey"
      columns: ["category_id"]
isOneToOne: false
      referencedRelation: "taxonomy_nodes"
      referencedColumns: ["id"]
    },{
      foreignKeyName: "products_import_batch_id_fkey"
      columns: ["import_batch_id"]
isOneToOne: false
      referencedRelation: "imports"
      referencedColumns: ["id"]
    }
                  ]
                },"regex_rules": {
                  Row: {
                    "category_id": string | null,"code": string,"confidence_score": number | null,"created_at": string | null,"created_by": string | null,"description": string | null,"flags": string | null,"id": string,"is_active": boolean | null,"name": string,"pattern": string,"priority": number | null,"test_cases": (string)[] | null,"updated_at": string | null,"updated_by": string | null
                  }
                  ComputedFields: never
                  Insert: {
                    "category_id"?: string | null,"code": string,"confidence_score"?: number | null,"created_at"?: string | null,"created_by"?: string | null,"description"?: string | null,"flags"?: string | null,"id"?: string,"is_active"?: boolean | null,"name": string,"pattern": string,"priority"?: number | null,"test_cases"?: (string)[] | null,"updated_at"?: string | null,"updated_by"?: string | null
                  }
                  Update: {
                    "category_id"?: string | null,"code"?: string,"confidence_score"?: number | null,"created_at"?: string | null,"created_by"?: string | null,"description"?: string | null,"flags"?: string | null,"id"?: string,"is_active"?: boolean | null,"name"?: string,"pattern"?: string,"priority"?: number | null,"test_cases"?: (string)[] | null,"updated_at"?: string | null,"updated_by"?: string | null
                  }
                  Relationships: [
                    {
      foreignKeyName: "regex_rules_category_id_fkey"
      columns: ["category_id"]
isOneToOne: false
      referencedRelation: "taxonomy_nodes"
      referencedColumns: ["id"]
    }
                  ]
                },"review_history": {
                  Row: {
                    "action": string,"comments": string | null,"created_at": string | null,"id": string,"metadata": Json | null,"new_category_id": string | null,"old_category_id": string | null,"product_id": string | null,"reviewer_id": string | null
                  }
                  ComputedFields: never
                  Insert: {
                    "action": string,"comments"?: string | null,"created_at"?: string | null,"id"?: string,"metadata"?: Json | null,"new_category_id"?: string | null,"old_category_id"?: string | null,"product_id"?: string | null,"reviewer_id"?: string | null
                  }
                  Update: {
                    "action"?: string,"comments"?: string | null,"created_at"?: string | null,"id"?: string,"metadata"?: Json | null,"new_category_id"?: string | null,"old_category_id"?: string | null,"product_id"?: string | null,"reviewer_id"?: string | null
                  }
                  Relationships: [
                    {
      foreignKeyName: "review_history_new_category_id_fkey"
      columns: ["new_category_id"]
isOneToOne: false
      referencedRelation: "taxonomy_nodes"
      referencedColumns: ["id"]
    },{
      foreignKeyName: "review_history_old_category_id_fkey"
      columns: ["old_category_id"]
isOneToOne: false
      referencedRelation: "taxonomy_nodes"
      referencedColumns: ["id"]
    },{
      foreignKeyName: "review_history_product_id_fkey"
      columns: ["product_id"]
isOneToOne: false
      referencedRelation: "products"
      referencedColumns: ["id"]
    }
                  ]
                },"similarity_matches": {
                  Row: {
                    "algorithm": string | null,"created_at": string | null,"id": string,"is_duplicate": boolean | null,"match_type": string | null,"metadata": Json | null,"product_a_id": string | null,"product_b_id": string | null,"reviewed": boolean | null,"reviewed_at": string | null,"reviewed_by": string | null,"similarity_score": number
                  }
                  ComputedFields: never
                  Insert: {
                    "algorithm"?: string | null,"created_at"?: string | null,"id"?: string,"is_duplicate"?: boolean | null,"match_type"?: string | null,"metadata"?: Json | null,"product_a_id"?: string | null,"product_b_id"?: string | null,"reviewed"?: boolean | null,"reviewed_at"?: string | null,"reviewed_by"?: string | null,"similarity_score": number
                  }
                  Update: {
                    "algorithm"?: string | null,"created_at"?: string | null,"id"?: string,"is_duplicate"?: boolean | null,"match_type"?: string | null,"metadata"?: Json | null,"product_a_id"?: string | null,"product_b_id"?: string | null,"reviewed"?: boolean | null,"reviewed_at"?: string | null,"reviewed_by"?: string | null,"similarity_score"?: number
                  }
                  Relationships: [
                    {
      foreignKeyName: "similarity_matches_product_a_id_fkey"
      columns: ["product_a_id"]
isOneToOne: false
      referencedRelation: "products"
      referencedColumns: ["id"]
    },{
      foreignKeyName: "similarity_matches_product_b_id_fkey"
      columns: ["product_b_id"]
isOneToOne: false
      referencedRelation: "products"
      referencedColumns: ["id"]
    }
                  ]
                },"synonym_lemmas": {
                  Row: {
                    "category_id": string | null,"code": string,"created_at": string | null,"created_by": string | null,"description": string | null,"id": string,"is_active": boolean | null,"is_verified": boolean | null,"name_en": string | null,"name_th": string,"updated_at": string | null,"updated_by": string | null
                  }
                  ComputedFields: never
                  Insert: {
                    "category_id"?: string | null,"code": string,"created_at"?: string | null,"created_by"?: string | null,"description"?: string | null,"id"?: string,"is_active"?: boolean | null,"is_verified"?: boolean | null,"name_en"?: string | null,"name_th": string,"updated_at"?: string | null,"updated_by"?: string | null
                  }
                  Update: {
                    "category_id"?: string | null,"code"?: string,"created_at"?: string | null,"created_by"?: string | null,"description"?: string | null,"id"?: string,"is_active"?: boolean | null,"is_verified"?: boolean | null,"name_en"?: string | null,"name_th"?: string,"updated_at"?: string | null,"updated_by"?: string | null
                  }
                  Relationships: [
                    {
      foreignKeyName: "synonym_lemmas_category_id_fkey"
      columns: ["category_id"]
isOneToOne: false
      referencedRelation: "taxonomy_nodes"
      referencedColumns: ["id"]
    }
                  ]
                },"synonym_terms": {
                  Row: {
                    "confidence_score": number | null,"created_at": string | null,"created_by": string | null,"id": string,"is_primary": boolean | null,"is_verified": boolean | null,"language": string | null,"lemma_id": string | null,"source": string | null,"term": string,"usage_count": number | null
                  }
                  ComputedFields: never
                  Insert: {
                    "confidence_score"?: number | null,"created_at"?: string | null,"created_by"?: string | null,"id"?: string,"is_primary"?: boolean | null,"is_verified"?: boolean | null,"language"?: string | null,"lemma_id"?: string | null,"source"?: string | null,"term": string,"usage_count"?: number | null
                  }
                  Update: {
                    "confidence_score"?: number | null,"created_at"?: string | null,"created_by"?: string | null,"id"?: string,"is_primary"?: boolean | null,"is_verified"?: boolean | null,"language"?: string | null,"lemma_id"?: string | null,"source"?: string | null,"term"?: string,"usage_count"?: number | null
                  }
                  Relationships: [
                    {
      foreignKeyName: "synonym_terms_lemma_id_fkey"
      columns: ["lemma_id"]
isOneToOne: false
      referencedRelation: "synonym_lemmas"
      referencedColumns: ["id"]
    }
                  ]
                },"system_settings": {
                  Row: {
                    "ai": Json | null,"description": string | null,"id": string,"processing": Json | null,"search": Json | null,"setting_key": string | null,"setting_value": string | null,"ui": Json | null,"updated_at": string | null,"updated_by": string | null
                  }
                  ComputedFields: never
                  Insert: {
                    "ai"?: Json | null,"description"?: string | null,"id"?: string,"processing"?: Json | null,"search"?: Json | null,"setting_key"?: string | null,"setting_value"?: string | null,"ui"?: Json | null,"updated_at"?: string | null,"updated_by"?: string | null
                  }
                  Update: {
                    "ai"?: Json | null,"description"?: string | null,"id"?: string,"processing"?: Json | null,"search"?: Json | null,"setting_key"?: string | null,"setting_value"?: string | null,"ui"?: Json | null,"updated_at"?: string | null,"updated_by"?: string | null
                  }
                  Relationships: [
                    
                  ]
                },"taxonomy_nodes": {
                  Row: {
                    "code": string,"created_at": string | null,"created_by": string | null,"description": string | null,"embedding": string | null,"id": string,"is_active": boolean | null,"keywords": (string)[] | null,"level": number | null,"metadata": Json | null,"name_en": string | null,"name_th": string,"parent_id": string | null,"path": string | null,"short_code": string | null,"sort_order": number | null,"updated_at": string | null,"updated_by": string | null
                  }
                  ComputedFields: never
                  Insert: {
                    "code": string,"created_at"?: string | null,"created_by"?: string | null,"description"?: string | null,"embedding"?: string | null,"id"?: string,"is_active"?: boolean | null,"keywords"?: (string)[] | null,"level"?: number | null,"metadata"?: Json | null,"name_en"?: string | null,"name_th": string,"parent_id"?: string | null,"path"?: string | null,"short_code"?: string | null,"sort_order"?: number | null,"updated_at"?: string | null,"updated_by"?: string | null
                  }
                  Update: {
                    "code"?: string,"created_at"?: string | null,"created_by"?: string | null,"description"?: string | null,"embedding"?: string | null,"id"?: string,"is_active"?: boolean | null,"keywords"?: (string)[] | null,"level"?: number | null,"metadata"?: Json | null,"name_en"?: string | null,"name_th"?: string,"parent_id"?: string | null,"path"?: string | null,"short_code"?: string | null,"sort_order"?: number | null,"updated_at"?: string | null,"updated_by"?: string | null
                  }
                  Relationships: [
                    {
      foreignKeyName: "taxonomy_nodes_parent_id_fkey"
      columns: ["parent_id"]
isOneToOne: false
      referencedRelation: "taxonomy_nodes"
      referencedColumns: ["id"]
    }
                  ]
                }
          }
          Views: {
            [_ in never]: never
          }
          Functions: {
            "batch_category_classification":
{ Args: { "product_data": Json,"top_k"?: number }; Returns: {
              "category_id": string,"category_name": string,"confidence": number,"method": string,"product_name": string
            }[]
                           },
"get_sample_categories_with_embeddings":
{ Args: { "sample_size"?: number }; Returns: {
              "embedding_dimension": number,"has_embedding": boolean,"id": string,"name_th": string
            }[]
                           },
"hybrid_category_classification":
{ Args: { "product_embedding": string,"product_name": string,"top_k"?: number }; Returns: {
              "category_id": string,"category_level": number,"category_name": string,"confidence": number,"matched_keyword": string,"method": string,"methods": (string)[]
            }[]
                           },
"match_categories_by_embedding":
{ Args: { "match_count"?: number,"match_threshold"?: number,"query_embedding": string }; Returns: {
              "category_id": string,"category_level": number,"category_name": string,"keywords": (string)[],"similarity": number
            }[]
                           },
"match_products_by_embedding":
{ Args: { "match_count"?: number,"match_threshold"?: number,"query_embedding": string }; Returns: {
              "category_id": string,"id": string,"name_th": string,"similarity": number,"status": string
            }[]
                           },
"recheck_agreement_stats":
{ Args: Record<PropertyKey, never>; Returns: {
              "agreed": number,"total": number
            }[]
                           }
          }
          Enums: {
            [_ in never]: never
          }
          CompositeTypes: {
            [_ in never]: never
          }
        }
}

type DatabaseWithoutInternals = Omit<Database, '__InternalSupabase'>

type DefaultSchema = DatabaseWithoutInternals[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never
> = DefaultSchemaTableNameOrOptions extends { schema: keyof DatabaseWithoutInternals }
  ? (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
  ? (DefaultSchema["Tables"] & DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
      Row: infer R
    }
    ? R
    : never
  : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never
> = DefaultSchemaTableNameOrOptions extends { schema: keyof DatabaseWithoutInternals }
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
  ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
      Insert: infer I
    }
    ? I
    : never
  : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never
> = DefaultSchemaTableNameOrOptions extends { schema: keyof DatabaseWithoutInternals }
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
  ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
      Update: infer U
    }
    ? U
    : never
  : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof DatabaseWithoutInternals },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never
> = DefaultSchemaEnumNameOrOptions extends { schema: keyof DatabaseWithoutInternals }
  ? DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
  ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
  : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof DatabaseWithoutInternals },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never
> = PublicCompositeTypeNameOrOptions extends { schema: keyof DatabaseWithoutInternals }
  ? DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
  ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
  : never

export const Constants = {
  "graphql_public": {
          Enums: {
            
          }
        },"public": {
          Enums: {
            
          }
        }
} as const
