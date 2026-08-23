


SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;


CREATE EXTENSION IF NOT EXISTS "pg_net" WITH SCHEMA "extensions";






COMMENT ON SCHEMA "public" IS 'standard public schema';



CREATE EXTENSION IF NOT EXISTS "pg_stat_statements" WITH SCHEMA "extensions";






CREATE EXTENSION IF NOT EXISTS "pgcrypto" WITH SCHEMA "extensions";






CREATE EXTENSION IF NOT EXISTS "supabase_vault" WITH SCHEMA "vault";






CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA "extensions";






CREATE EXTENSION IF NOT EXISTS "vector" WITH SCHEMA "public";






CREATE OR REPLACE FUNCTION "public"."audit_trigger_function"() RETURNS "trigger"
    LANGUAGE "plpgsql"
    AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        INSERT INTO audit_logs (table_name, record_id, action, old_values, user_id)
        VALUES (TG_TABLE_NAME, OLD.id::TEXT, 'DELETE', row_to_json(OLD), current_setting('app.current_user_id', true)::UUID);
        RETURN OLD;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_logs (table_name, record_id, action, old_values, new_values, user_id)
        VALUES (TG_TABLE_NAME, NEW.id::TEXT, 'UPDATE', row_to_json(OLD), row_to_json(NEW), current_setting('app.current_user_id', true)::UUID);
        RETURN NEW;
    ELSIF TG_OP = 'INSERT' THEN
        INSERT INTO audit_logs (table_name, record_id, action, new_values, user_id)
        VALUES (TG_TABLE_NAME, NEW.id::TEXT, 'INSERT', row_to_json(NEW), current_setting('app.current_user_id', true)::UUID);
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$;


ALTER FUNCTION "public"."audit_trigger_function"() OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."batch_category_classification"("product_data" "jsonb", "top_k" integer DEFAULT 3) RETURNS TABLE("product_name" "text", "category_id" "uuid", "category_name" "text", "confidence" double precision, "method" "text")
    LANGUAGE "plpgsql" STABLE
    AS $$
DECLARE
  product jsonb;
BEGIN
  FOR product IN SELECT * FROM jsonb_array_elements(product_data)
  LOOP
    RETURN QUERY
    SELECT 
      product->>'name' as product_name,
      h.category_id,
      h.category_name,
      h.confidence,
      h.method
    FROM hybrid_category_classification(
      product->>'name',
      (product->'embedding')::vector(384),
      1  -- top 1 only for batch
    ) h
    LIMIT 1;
  END LOOP;
END;
$$;


ALTER FUNCTION "public"."batch_category_classification"("product_data" "jsonb", "top_k" integer) OWNER TO "postgres";


COMMENT ON FUNCTION "public"."batch_category_classification"("product_data" "jsonb", "top_k" integer) IS 'Batch classify multiple products (optimized for performance, 384-dim)';



CREATE OR REPLACE FUNCTION "public"."exec_sql"("query_text" "text", "query_params" "jsonb" DEFAULT '[]'::"jsonb") RETURNS "jsonb"
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
    AS $$
DECLARE
    result JSONB;
    row_count INTEGER;
BEGIN
    -- Execute dynamic SQL with parameters
    IF query_params IS NOT NULL AND jsonb_array_length(query_params) > 0 THEN
        -- For parameterized queries (more complex implementation would be needed)
        EXECUTE query_text;
    ELSE
        EXECUTE query_text;
    END IF;
    
    GET DIAGNOSTICS row_count = ROW_COUNT;
    
    -- Return basic execution info
    result = jsonb_build_object(
        'success', true,
        'rows_affected', row_count,
        'executed_at', now()
    );
    
    RETURN result;
EXCEPTION
    WHEN OTHERS THEN
        RETURN jsonb_build_object(
            'success', false,
            'error', SQLERRM,
            'error_code', SQLSTATE
        );
END;
$$;


ALTER FUNCTION "public"."exec_sql"("query_text" "text", "query_params" "jsonb") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."get_sample_categories_with_embeddings"("sample_size" integer DEFAULT 5) RETURNS TABLE("id" "uuid", "name_th" "text", "has_embedding" boolean, "embedding_dimension" integer)
    LANGUAGE "sql" STABLE
    AS $$
  SELECT 
    id,
    name_th,
    embedding IS NOT NULL as has_embedding,
    384 as embedding_dimension
  FROM taxonomy_nodes
  WHERE is_active = true
  ORDER BY created_at DESC
  LIMIT sample_size;
$$;


ALTER FUNCTION "public"."get_sample_categories_with_embeddings"("sample_size" integer) OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."hybrid_category_classification"("product_name" "text", "product_embedding" "public"."vector", "top_k" integer DEFAULT 3) RETURNS TABLE("category_id" "uuid", "category_name" "text", "category_level" integer, "confidence" double precision, "method" "text", "matched_keyword" "text", "methods" "text"[])
    LANGUAGE "plpgsql" STABLE
    AS $$
DECLARE
  keyword_weight float := 0.6;  -- 60% weight for keyword matching
  embedding_weight float := 0.4; -- 40% weight for embedding similarity
BEGIN
  RETURN QUERY
  WITH keyword_matches AS (
    -- 1. Keyword rule matching
    SELECT 
      kr.category_id,
      tn.name_th as category_name,
      tn.level as category_level,
      kr.confidence_score * keyword_weight as confidence,
      'keyword_rule' as method,
      unnest(kr.keywords) as matched_keyword,
      ARRAY['keyword_rule']::text[] as methods
    FROM keyword_rules kr
    JOIN taxonomy_nodes tn ON kr.category_id = tn.id
    WHERE 
      kr.is_active = true
      AND tn.is_active = true
      AND EXISTS (
        SELECT 1 FROM unnest(kr.keywords) kw
        WHERE product_name ILIKE '%' || kw || '%'
      )
    
    UNION
    
    -- 2. Taxonomy keyword matching
    SELECT 
      tn.id as category_id,
      tn.name_th as category_name,
      tn.level as category_level,
      0.7 * keyword_weight as confidence,
      'taxonomy_keyword' as method,
      (
        SELECT kw FROM unnest(tn.keywords) kw 
        WHERE product_name ILIKE '%' || kw || '%' 
        LIMIT 1
      ) as matched_keyword,
      ARRAY['taxonomy_keyword']::text[] as methods
    FROM taxonomy_nodes tn
    WHERE 
      tn.is_active = true
      AND tn.keywords IS NOT NULL
      AND EXISTS (
        SELECT 1 FROM unnest(tn.keywords) kw
        WHERE product_name ILIKE '%' || kw || '%'
      )
    
    UNION
    
    -- 3. Category name matching
    SELECT 
      tn.id as category_id,
      tn.name_th as category_name,
      tn.level as category_level,
      0.95 * keyword_weight as confidence,
      'name_match' as method,
      tn.name_th as matched_keyword,
      ARRAY['name_match']::text[] as methods
    FROM taxonomy_nodes tn
    WHERE 
      tn.is_active = true
      AND product_name ILIKE '%' || tn.name_th || '%'
  ),
  
  embedding_matches AS (
    -- 4. Embedding similarity matching
    SELECT 
      tn.id as category_id,
      tn.name_th as category_name,
      tn.level as category_level,
      (1 - (tn.embedding <=> product_embedding)) * embedding_weight as confidence,
      'embedding' as method,
      NULL::text as matched_keyword,
      ARRAY['embedding']::text[] as methods
    FROM taxonomy_nodes tn
    WHERE 
      tn.embedding IS NOT NULL
      AND tn.is_active = true
      AND (1 - (tn.embedding <=> product_embedding)) >= 0.3
    ORDER BY tn.embedding <=> product_embedding
    LIMIT 10
  ),
  
  combined AS (
    -- Combine keyword and embedding matches
    SELECT 
      category_id,
      category_name,
      category_level,
      SUM(confidence) as total_confidence,
      MAX(method) as primary_method,
      MAX(matched_keyword) as matched_keyword,
      array_agg(DISTINCT m ORDER BY m) as methods
    FROM (
      SELECT 
        category_id,
        category_name,
        category_level,
        confidence,
        method,
        matched_keyword,
        unnest(methods) as m
      FROM keyword_matches
      
      UNION ALL
      
      SELECT 
        category_id,
        category_name,
        category_level,
        confidence,
        method,
        matched_keyword,
        unnest(methods) as m
      FROM embedding_matches
    ) all_matches
    GROUP BY category_id, category_name, category_level
  )
  
  SELECT 
    c.category_id,
    c.category_name,
    c.category_level,
    LEAST(c.total_confidence, 0.99) as confidence, -- Cap at 0.99
    CASE 
      WHEN array_length(c.methods, 1) > 1 THEN 'hybrid'
      ELSE c.primary_method
    END as method,
    c.matched_keyword,
    c.methods
  FROM combined c
  ORDER BY c.total_confidence DESC, c.category_level ASC
  LIMIT top_k;
END;
$$;


ALTER FUNCTION "public"."hybrid_category_classification"("product_name" "text", "product_embedding" "public"."vector", "top_k" integer) OWNER TO "postgres";


COMMENT ON FUNCTION "public"."hybrid_category_classification"("product_name" "text", "product_embedding" "public"."vector", "top_k" integer) IS 'Hybrid category classification: 60% keyword matching + 40% embedding similarity (384-dim, 72% accuracy)';



CREATE OR REPLACE FUNCTION "public"."match_categories_by_embedding"("query_embedding" "public"."vector", "match_threshold" double precision DEFAULT 0.5, "match_count" integer DEFAULT 5) RETURNS TABLE("category_id" "uuid", "category_name" "text", "category_level" integer, "similarity" double precision, "keywords" "text"[])
    LANGUAGE "plpgsql" STABLE
    AS $$
BEGIN
  RETURN QUERY
  SELECT 
    tn.id,
    tn.name_th,
    tn.level,
    1 - (tn.embedding <=> query_embedding) as similarity,
    tn.keywords
  FROM taxonomy_nodes tn
  WHERE 
    tn.embedding IS NOT NULL
    AND tn.is_active = true
    AND (1 - (tn.embedding <=> query_embedding)) >= match_threshold
  ORDER BY tn.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;


ALTER FUNCTION "public"."match_categories_by_embedding"("query_embedding" "public"."vector", "match_threshold" double precision, "match_count" integer) OWNER TO "postgres";


COMMENT ON FUNCTION "public"."match_categories_by_embedding"("query_embedding" "public"."vector", "match_threshold" double precision, "match_count" integer) IS 'Vector similarity search for category matching using pgvector cosine distance (384-dim)';



CREATE OR REPLACE FUNCTION "public"."update_updated_at_column"() RETURNS "trigger"
    LANGUAGE "plpgsql"
    AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;


ALTER FUNCTION "public"."update_updated_at_column"() OWNER TO "postgres";

SET default_tablespace = '';

SET default_table_access_method = "heap";


CREATE TABLE IF NOT EXISTS "public"."audit_logs" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "table_name" "text" NOT NULL,
    "record_id" "text" NOT NULL,
    "action" "text" NOT NULL,
    "old_values" "jsonb",
    "new_values" "jsonb",
    "changed_fields" "text"[],
    "user_id" "uuid",
    "user_agent" "text",
    "ip_address" "inet",
    "session_id" "text",
    "created_at" timestamp with time zone DEFAULT "now"(),
    CONSTRAINT "audit_logs_action_check" CHECK (("action" = ANY (ARRAY['INSERT'::"text", 'UPDATE'::"text", 'DELETE'::"text"])))
);


ALTER TABLE "public"."audit_logs" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."human_feedback" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "old_product" "text" NOT NULL,
    "new_product" "text" NOT NULL,
    "similarity_score" double precision NOT NULL,
    "human_decision" "text" NOT NULL,
    "ml_prediction" "text",
    "reviewer_id" "uuid",
    "comments" "text",
    "confidence_score" double precision DEFAULT 0.0,
    "processing_time" integer DEFAULT 0,
    "created_at" timestamp with time zone DEFAULT "now"(),
    "updated_at" timestamp with time zone DEFAULT "now"(),
    CONSTRAINT "human_feedback_human_decision_check" CHECK (("human_decision" = ANY (ARRAY['similar'::"text", 'different'::"text", 'duplicate'::"text", 'uncertain'::"text"]))),
    CONSTRAINT "human_feedback_ml_prediction_check" CHECK (("ml_prediction" = ANY (ARRAY['similar'::"text", 'different'::"text"])))
);


ALTER TABLE "public"."human_feedback" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."imports" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "name" "text" NOT NULL,
    "description" "text",
    "file_name" "text",
    "file_size" bigint,
    "file_type" "text",
    "total_records" integer DEFAULT 0,
    "processed_records" integer DEFAULT 0,
    "success_records" integer DEFAULT 0,
    "error_records" integer DEFAULT 0,
    "status" "text" DEFAULT 'pending'::"text",
    "error_details" "jsonb",
    "metadata" "jsonb" DEFAULT '{}'::"jsonb",
    "created_by" "uuid",
    "started_at" timestamp with time zone,
    "completed_at" timestamp with time zone,
    "created_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."imports" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."keyword_rules" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "code" "text" NOT NULL,
    "name" "text" NOT NULL,
    "description" "text",
    "keywords" "text"[] NOT NULL,
    "category_id" "uuid",
    "priority" integer DEFAULT 0,
    "match_type" "text" DEFAULT 'contains'::"text",
    "confidence_score" double precision DEFAULT 0.8,
    "is_active" boolean DEFAULT true,
    "created_by" "uuid",
    "updated_by" "uuid",
    "created_at" timestamp with time zone DEFAULT "now"(),
    "updated_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."keyword_rules" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."ml_training_history" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "training_date" timestamp with time zone DEFAULT "now"() NOT NULL,
    "model_type" "text" DEFAULT 'random_forest'::"text" NOT NULL,
    "total_samples" integer DEFAULT 0 NOT NULL,
    "train_samples" integer DEFAULT 0 NOT NULL,
    "test_samples" integer DEFAULT 0 NOT NULL,
    "train_accuracy" double precision DEFAULT 0 NOT NULL,
    "test_accuracy" double precision DEFAULT 0 NOT NULL,
    "cv_mean_accuracy" double precision DEFAULT 0,
    "cv_std_accuracy" double precision DEFAULT 0,
    "feature_importance" "jsonb" DEFAULT '[]'::"jsonb",
    "classes" "jsonb" DEFAULT '[]'::"jsonb",
    "classification_report" "jsonb" DEFAULT '{}'::"jsonb",
    "created_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."ml_training_history" OWNER TO "postgres";


COMMENT ON TABLE "public"."ml_training_history" IS 'เก็บประวัติการเทรนโมเดล ML ทุก session — สร้างโดย Phayak 2026-05-30';



COMMENT ON COLUMN "public"."ml_training_history"."feature_importance" IS 'Array ของ {feature: string, importance: float} เรียงตามความสำคัญ';



COMMENT ON COLUMN "public"."ml_training_history"."classes" IS 'รายการ class labels ที่โมเดลรู้จัก เช่น ["duplicate","different"]';



CREATE TABLE IF NOT EXISTS "public"."product_attributes" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "product_id" "uuid",
    "attribute_name" "text" NOT NULL,
    "attribute_value" "text" NOT NULL,
    "attribute_type" "text" DEFAULT 'text'::"text",
    "created_by" "uuid",
    "created_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."product_attributes" OWNER TO "postgres";


COMMENT ON COLUMN "public"."product_attributes"."product_id" IS 'Optional reference to products table. Can be NULL for standalone attributes.';



CREATE TABLE IF NOT EXISTS "public"."product_category_suggestions" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "product_id" "uuid",
    "suggested_category_id" "uuid",
    "confidence_score" double precision NOT NULL,
    "suggestion_method" "text" NOT NULL,
    "rule_id" "uuid",
    "metadata" "jsonb" DEFAULT '{}'::"jsonb",
    "is_accepted" boolean,
    "reviewed_by" "uuid",
    "reviewed_at" timestamp with time zone,
    "created_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."product_category_suggestions" OWNER TO "postgres";


COMMENT ON COLUMN "public"."product_category_suggestions"."product_id" IS 'Optional reference to products table. Can be NULL for standalone suggestions.';



CREATE TABLE IF NOT EXISTS "public"."products" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "name_th" "text" NOT NULL,
    "name_en" "text",
    "description" "text",
    "category_id" "uuid",
    "brand" "text",
    "model" "text",
    "sku" "text",
    "price" numeric(12,2),
    "embedding" "public"."vector"(768),
    "keywords" "text"[],
    "metadata" "jsonb" DEFAULT '{}'::"jsonb",
    "status" "text" DEFAULT 'pending'::"text",
    "confidence_score" double precision,
    "import_batch_id" "uuid",
    "reviewed_by" "uuid",
    "reviewed_at" timestamp with time zone,
    "created_by" "uuid",
    "updated_by" "uuid",
    "created_at" timestamp with time zone DEFAULT "now"(),
    "updated_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."products" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."regex_rules" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "code" "text" NOT NULL,
    "name" "text" NOT NULL,
    "description" "text",
    "pattern" "text" NOT NULL,
    "flags" "text" DEFAULT 'gi'::"text",
    "category_id" "uuid",
    "priority" integer DEFAULT 0,
    "confidence_score" numeric(3,2) DEFAULT 0.7,
    "is_active" boolean DEFAULT true,
    "test_cases" "text"[] DEFAULT '{}'::"text"[],
    "created_by" "uuid",
    "updated_by" "uuid",
    "created_at" timestamp with time zone DEFAULT "now"(),
    "updated_at" timestamp with time zone DEFAULT "now"(),
    CONSTRAINT "regex_rules_confidence_score_check" CHECK ((("confidence_score" >= (0)::numeric) AND ("confidence_score" <= (1)::numeric)))
);


ALTER TABLE "public"."regex_rules" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."review_history" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "product_id" "uuid",
    "reviewer_id" "uuid",
    "action" "text" NOT NULL,
    "old_category_id" "uuid",
    "new_category_id" "uuid",
    "comments" "text",
    "metadata" "jsonb" DEFAULT '{}'::"jsonb",
    "created_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."review_history" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."similarity_matches" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "product_a_id" "uuid",
    "product_b_id" "uuid",
    "similarity_score" double precision NOT NULL,
    "match_type" "text" DEFAULT 'semantic'::"text",
    "algorithm" "text" DEFAULT 'cosine'::"text",
    "is_duplicate" boolean DEFAULT false,
    "reviewed" boolean DEFAULT false,
    "reviewed_by" "uuid",
    "reviewed_at" timestamp with time zone,
    "metadata" "jsonb" DEFAULT '{}'::"jsonb",
    "created_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."similarity_matches" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."synonym_lemmas" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "code" "text" NOT NULL,
    "name_th" "text" NOT NULL,
    "name_en" "text",
    "description" "text",
    "category_id" "uuid",
    "is_verified" boolean DEFAULT false,
    "is_active" boolean DEFAULT true,
    "created_by" "uuid",
    "updated_by" "uuid",
    "created_at" timestamp with time zone DEFAULT "now"(),
    "updated_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."synonym_lemmas" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."synonym_terms" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "lemma_id" "uuid",
    "term" "text" NOT NULL,
    "is_primary" boolean DEFAULT false,
    "confidence_score" double precision DEFAULT 1.0,
    "usage_count" integer DEFAULT 0,
    "source" "text" DEFAULT 'manual'::"text",
    "language" "text" DEFAULT 'th'::"text",
    "is_verified" boolean DEFAULT false,
    "created_by" "uuid",
    "created_at" timestamp with time zone DEFAULT "now"()
);


ALTER TABLE "public"."synonym_terms" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."system_settings" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "search" "jsonb" DEFAULT '{"maxResults": 50, "defaultSearchType": "hybrid", "textSearchEnabled": true, "confidenceThreshold": 0.5, "hybridSearchEnabled": true, "vectorSearchEnabled": true}'::"jsonb",
    "processing" "jsonb" DEFAULT '{"batchSize": 100, "retryAttempts": 3, "timeoutSeconds": 30, "maxConcurrentJobs": 5}'::"jsonb",
    "ai" "jsonb" DEFAULT '{"maxTokens": 4000, "apiProvider": "openai", "temperature": 0.1, "embeddingModel": "text-embedding-ada-002"}'::"jsonb",
    "ui" "jsonb" DEFAULT '{"theme": "light", "language": "th", "itemsPerPage": 20, "enableAnimations": true}'::"jsonb",
    "updated_by" "uuid",
    "updated_at" timestamp with time zone DEFAULT "now"(),
    "setting_key" "text",
    "setting_value" "text",
    "description" "text"
);


ALTER TABLE "public"."system_settings" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."taxonomy_nodes" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "code" "text" NOT NULL,
    "name_th" "text" NOT NULL,
    "name_en" "text",
    "description" "text",
    "parent_id" "uuid",
    "level" integer DEFAULT 0,
    "sort_order" integer DEFAULT 0,
    "path" "text",
    "keywords" "text"[],
    "metadata" "jsonb" DEFAULT '{}'::"jsonb",
    "is_active" boolean DEFAULT true,
    "created_by" "uuid",
    "updated_by" "uuid",
    "created_at" timestamp with time zone DEFAULT "now"(),
    "updated_at" timestamp with time zone DEFAULT "now"(),
    "short_code" "text",
    "embedding" "public"."vector"(384)
);


ALTER TABLE "public"."taxonomy_nodes" OWNER TO "postgres";


COMMENT ON COLUMN "public"."taxonomy_nodes"."embedding" IS 'Vector embedding for category name (384-dim for local model paraphrase-multilingual-MiniLM-L12-v2)';



ALTER TABLE ONLY "public"."audit_logs"
    ADD CONSTRAINT "audit_logs_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."human_feedback"
    ADD CONSTRAINT "human_feedback_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."imports"
    ADD CONSTRAINT "imports_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."keyword_rules"
    ADD CONSTRAINT "keyword_rules_code_key" UNIQUE ("code");



ALTER TABLE ONLY "public"."keyword_rules"
    ADD CONSTRAINT "keyword_rules_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."ml_training_history"
    ADD CONSTRAINT "ml_training_history_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."product_attributes"
    ADD CONSTRAINT "product_attributes_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."product_category_suggestions"
    ADD CONSTRAINT "product_category_suggestions_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."products"
    ADD CONSTRAINT "products_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."products"
    ADD CONSTRAINT "products_sku_key" UNIQUE ("sku");



ALTER TABLE ONLY "public"."regex_rules"
    ADD CONSTRAINT "regex_rules_code_key" UNIQUE ("code");



ALTER TABLE ONLY "public"."regex_rules"
    ADD CONSTRAINT "regex_rules_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."review_history"
    ADD CONSTRAINT "review_history_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."similarity_matches"
    ADD CONSTRAINT "similarity_matches_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."similarity_matches"
    ADD CONSTRAINT "similarity_matches_product_a_id_product_b_id_key" UNIQUE ("product_a_id", "product_b_id");



ALTER TABLE ONLY "public"."synonym_lemmas"
    ADD CONSTRAINT "synonym_lemmas_code_key" UNIQUE ("code");



ALTER TABLE ONLY "public"."synonym_lemmas"
    ADD CONSTRAINT "synonym_lemmas_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."synonym_terms"
    ADD CONSTRAINT "synonym_terms_lemma_id_term_key" UNIQUE ("lemma_id", "term");



ALTER TABLE ONLY "public"."synonym_terms"
    ADD CONSTRAINT "synonym_terms_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."system_settings"
    ADD CONSTRAINT "system_settings_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."system_settings"
    ADD CONSTRAINT "system_settings_setting_key_key" UNIQUE ("setting_key");



ALTER TABLE ONLY "public"."taxonomy_nodes"
    ADD CONSTRAINT "taxonomy_nodes_code_key" UNIQUE ("code");



ALTER TABLE ONLY "public"."taxonomy_nodes"
    ADD CONSTRAINT "taxonomy_nodes_pkey" PRIMARY KEY ("id");



CREATE INDEX "idx_audit_logs_action" ON "public"."audit_logs" USING "btree" ("action");



CREATE INDEX "idx_audit_logs_created_at" ON "public"."audit_logs" USING "btree" ("created_at" DESC);



CREATE INDEX "idx_audit_logs_table_record" ON "public"."audit_logs" USING "btree" ("table_name", "record_id");



CREATE INDEX "idx_audit_logs_user_id" ON "public"."audit_logs" USING "btree" ("user_id") WHERE ("user_id" IS NOT NULL);



CREATE INDEX "idx_human_feedback_created_at" ON "public"."human_feedback" USING "btree" ("created_at" DESC);



CREATE INDEX "idx_human_feedback_decision" ON "public"."human_feedback" USING "btree" ("human_decision");



CREATE INDEX "idx_human_feedback_products" ON "public"."human_feedback" USING "btree" ("old_product", "new_product");



CREATE INDEX "idx_human_feedback_reviewer" ON "public"."human_feedback" USING "btree" ("reviewer_id") WHERE ("reviewer_id" IS NOT NULL);



CREATE INDEX "idx_imports_created_at" ON "public"."imports" USING "btree" ("created_at" DESC);



CREATE INDEX "idx_imports_created_by" ON "public"."imports" USING "btree" ("created_by");



CREATE INDEX "idx_imports_status" ON "public"."imports" USING "btree" ("status");



CREATE INDEX "idx_keyword_rules_active" ON "public"."keyword_rules" USING "btree" ("is_active") WHERE ("is_active" = true);



CREATE INDEX "idx_keyword_rules_category" ON "public"."keyword_rules" USING "btree" ("category_id");



CREATE INDEX "idx_keyword_rules_code" ON "public"."keyword_rules" USING "btree" ("code");



CREATE INDEX "idx_keyword_rules_keywords" ON "public"."keyword_rules" USING "gin" ("keywords");



CREATE INDEX "idx_keyword_rules_priority" ON "public"."keyword_rules" USING "btree" ("priority" DESC);



CREATE INDEX "idx_ml_training_history_accuracy" ON "public"."ml_training_history" USING "btree" ("test_accuracy" DESC);



CREATE INDEX "idx_ml_training_history_date" ON "public"."ml_training_history" USING "btree" ("training_date" DESC);



CREATE INDEX "idx_product_attributes_name" ON "public"."product_attributes" USING "btree" ("attribute_name");



CREATE INDEX "idx_product_attributes_product_id" ON "public"."product_attributes" USING "btree" ("product_id");



CREATE INDEX "idx_product_suggestions_accepted" ON "public"."product_category_suggestions" USING "btree" ("is_accepted") WHERE ("is_accepted" IS NOT NULL);



CREATE INDEX "idx_product_suggestions_category" ON "public"."product_category_suggestions" USING "btree" ("suggested_category_id");



CREATE INDEX "idx_product_suggestions_confidence" ON "public"."product_category_suggestions" USING "btree" ("confidence_score" DESC);



CREATE INDEX "idx_product_suggestions_method" ON "public"."product_category_suggestions" USING "btree" ("suggestion_method");



CREATE INDEX "idx_product_suggestions_product" ON "public"."product_category_suggestions" USING "btree" ("product_id");



CREATE INDEX "idx_products_brand" ON "public"."products" USING "btree" ("brand") WHERE ("brand" IS NOT NULL);



CREATE INDEX "idx_products_category_id" ON "public"."products" USING "btree" ("category_id");



CREATE INDEX "idx_products_created_at" ON "public"."products" USING "btree" ("created_at");



CREATE INDEX "idx_products_embedding" ON "public"."products" USING "ivfflat" ("embedding" "public"."vector_cosine_ops") WITH ("lists"='100');



CREATE INDEX "idx_products_import_batch" ON "public"."products" USING "btree" ("import_batch_id") WHERE ("import_batch_id" IS NOT NULL);



CREATE INDEX "idx_products_keywords" ON "public"."products" USING "gin" ("keywords");



CREATE INDEX "idx_products_sku" ON "public"."products" USING "btree" ("sku") WHERE ("sku" IS NOT NULL);



CREATE INDEX "idx_products_status" ON "public"."products" USING "btree" ("status");



CREATE INDEX "idx_regex_rules_active" ON "public"."regex_rules" USING "btree" ("is_active") WHERE ("is_active" = true);



CREATE INDEX "idx_regex_rules_category" ON "public"."regex_rules" USING "btree" ("category_id");



CREATE INDEX "idx_regex_rules_code" ON "public"."regex_rules" USING "btree" ("code");



CREATE INDEX "idx_regex_rules_priority" ON "public"."regex_rules" USING "btree" ("priority" DESC);



CREATE INDEX "idx_review_history_action" ON "public"."review_history" USING "btree" ("action");



CREATE INDEX "idx_review_history_created_at" ON "public"."review_history" USING "btree" ("created_at" DESC);



CREATE INDEX "idx_review_history_product_id" ON "public"."review_history" USING "btree" ("product_id");



CREATE INDEX "idx_review_history_reviewer" ON "public"."review_history" USING "btree" ("reviewer_id") WHERE ("reviewer_id" IS NOT NULL);



CREATE INDEX "idx_similarity_matches_duplicate" ON "public"."similarity_matches" USING "btree" ("is_duplicate") WHERE ("is_duplicate" = true);



CREATE INDEX "idx_similarity_matches_product_a" ON "public"."similarity_matches" USING "btree" ("product_a_id");



CREATE INDEX "idx_similarity_matches_product_b" ON "public"."similarity_matches" USING "btree" ("product_b_id");



CREATE INDEX "idx_similarity_matches_reviewed" ON "public"."similarity_matches" USING "btree" ("reviewed");



CREATE INDEX "idx_similarity_matches_score" ON "public"."similarity_matches" USING "btree" ("similarity_score" DESC);



CREATE INDEX "idx_similarity_matches_type" ON "public"."similarity_matches" USING "btree" ("match_type");



CREATE INDEX "idx_synonym_lemmas_active" ON "public"."synonym_lemmas" USING "btree" ("is_active") WHERE ("is_active" = true);



CREATE INDEX "idx_synonym_lemmas_category_id" ON "public"."synonym_lemmas" USING "btree" ("category_id");



CREATE INDEX "idx_synonym_lemmas_code" ON "public"."synonym_lemmas" USING "btree" ("code");



CREATE INDEX "idx_synonym_terms_lemma_id" ON "public"."synonym_terms" USING "btree" ("lemma_id");



CREATE INDEX "idx_synonym_terms_primary" ON "public"."synonym_terms" USING "btree" ("is_primary") WHERE ("is_primary" = true);



CREATE INDEX "idx_synonym_terms_term" ON "public"."synonym_terms" USING "btree" ("term");



CREATE INDEX "idx_synonym_terms_verified" ON "public"."synonym_terms" USING "btree" ("is_verified");



CREATE INDEX "idx_taxonomy_nodes_active" ON "public"."taxonomy_nodes" USING "btree" ("is_active") WHERE ("is_active" = true);



CREATE INDEX "idx_taxonomy_nodes_code" ON "public"."taxonomy_nodes" USING "btree" ("code");



CREATE INDEX "idx_taxonomy_nodes_embedding" ON "public"."taxonomy_nodes" USING "ivfflat" ("embedding" "public"."vector_cosine_ops") WITH ("lists"='100');



CREATE INDEX "idx_taxonomy_nodes_embedding_vector" ON "public"."taxonomy_nodes" USING "ivfflat" ("embedding" "public"."vector_cosine_ops") WITH ("lists"='100');



CREATE INDEX "idx_taxonomy_nodes_keywords" ON "public"."taxonomy_nodes" USING "gin" ("keywords");



CREATE INDEX "idx_taxonomy_nodes_keywords_gin" ON "public"."taxonomy_nodes" USING "gin" ("keywords");



CREATE INDEX "idx_taxonomy_nodes_level" ON "public"."taxonomy_nodes" USING "btree" ("level");



CREATE INDEX "idx_taxonomy_nodes_parent_id" ON "public"."taxonomy_nodes" USING "btree" ("parent_id");



CREATE INDEX "idx_taxonomy_nodes_path" ON "public"."taxonomy_nodes" USING "btree" ("path");



CREATE INDEX "idx_taxonomy_nodes_short_code" ON "public"."taxonomy_nodes" USING "btree" ("short_code") WHERE ("short_code" IS NOT NULL);



CREATE OR REPLACE TRIGGER "audit_human_feedback" AFTER INSERT OR DELETE OR UPDATE ON "public"."human_feedback" FOR EACH ROW EXECUTE FUNCTION "public"."audit_trigger_function"();



CREATE OR REPLACE TRIGGER "audit_keyword_rules" AFTER INSERT OR DELETE OR UPDATE ON "public"."keyword_rules" FOR EACH ROW EXECUTE FUNCTION "public"."audit_trigger_function"();



CREATE OR REPLACE TRIGGER "audit_products" AFTER INSERT OR DELETE OR UPDATE ON "public"."products" FOR EACH ROW EXECUTE FUNCTION "public"."audit_trigger_function"();



CREATE OR REPLACE TRIGGER "audit_synonym_lemmas" AFTER INSERT OR DELETE OR UPDATE ON "public"."synonym_lemmas" FOR EACH ROW EXECUTE FUNCTION "public"."audit_trigger_function"();



CREATE OR REPLACE TRIGGER "audit_taxonomy_nodes" AFTER INSERT OR DELETE OR UPDATE ON "public"."taxonomy_nodes" FOR EACH ROW EXECUTE FUNCTION "public"."audit_trigger_function"();



CREATE OR REPLACE TRIGGER "update_human_feedback_updated_at" BEFORE UPDATE ON "public"."human_feedback" FOR EACH ROW EXECUTE FUNCTION "public"."update_updated_at_column"();



CREATE OR REPLACE TRIGGER "update_keyword_rules_updated_at" BEFORE UPDATE ON "public"."keyword_rules" FOR EACH ROW EXECUTE FUNCTION "public"."update_updated_at_column"();



CREATE OR REPLACE TRIGGER "update_products_updated_at" BEFORE UPDATE ON "public"."products" FOR EACH ROW EXECUTE FUNCTION "public"."update_updated_at_column"();



CREATE OR REPLACE TRIGGER "update_regex_rules_updated_at" BEFORE UPDATE ON "public"."regex_rules" FOR EACH ROW EXECUTE FUNCTION "public"."update_updated_at_column"();



CREATE OR REPLACE TRIGGER "update_synonym_lemmas_updated_at" BEFORE UPDATE ON "public"."synonym_lemmas" FOR EACH ROW EXECUTE FUNCTION "public"."update_updated_at_column"();



CREATE OR REPLACE TRIGGER "update_taxonomy_nodes_updated_at" BEFORE UPDATE ON "public"."taxonomy_nodes" FOR EACH ROW EXECUTE FUNCTION "public"."update_updated_at_column"();



ALTER TABLE ONLY "public"."audit_logs"
    ADD CONSTRAINT "audit_logs_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "auth"."users"("id");



ALTER TABLE ONLY "public"."human_feedback"
    ADD CONSTRAINT "human_feedback_reviewer_id_fkey" FOREIGN KEY ("reviewer_id") REFERENCES "auth"."users"("id");



ALTER TABLE ONLY "public"."keyword_rules"
    ADD CONSTRAINT "keyword_rules_category_id_fkey" FOREIGN KEY ("category_id") REFERENCES "public"."taxonomy_nodes"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."product_attributes"
    ADD CONSTRAINT "product_attributes_product_id_fkey" FOREIGN KEY ("product_id") REFERENCES "public"."products"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."product_category_suggestions"
    ADD CONSTRAINT "product_category_suggestions_product_id_fkey" FOREIGN KEY ("product_id") REFERENCES "public"."products"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."product_category_suggestions"
    ADD CONSTRAINT "product_category_suggestions_rule_id_fkey" FOREIGN KEY ("rule_id") REFERENCES "public"."keyword_rules"("id") ON DELETE SET NULL;



ALTER TABLE ONLY "public"."product_category_suggestions"
    ADD CONSTRAINT "product_category_suggestions_suggested_category_id_fkey" FOREIGN KEY ("suggested_category_id") REFERENCES "public"."taxonomy_nodes"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."products"
    ADD CONSTRAINT "products_category_id_fkey" FOREIGN KEY ("category_id") REFERENCES "public"."taxonomy_nodes"("id");



ALTER TABLE ONLY "public"."regex_rules"
    ADD CONSTRAINT "regex_rules_category_id_fkey" FOREIGN KEY ("category_id") REFERENCES "public"."taxonomy_nodes"("id");



ALTER TABLE ONLY "public"."regex_rules"
    ADD CONSTRAINT "regex_rules_created_by_fkey" FOREIGN KEY ("created_by") REFERENCES "auth"."users"("id");



ALTER TABLE ONLY "public"."regex_rules"
    ADD CONSTRAINT "regex_rules_updated_by_fkey" FOREIGN KEY ("updated_by") REFERENCES "auth"."users"("id");



ALTER TABLE ONLY "public"."review_history"
    ADD CONSTRAINT "review_history_new_category_id_fkey" FOREIGN KEY ("new_category_id") REFERENCES "public"."taxonomy_nodes"("id");



ALTER TABLE ONLY "public"."review_history"
    ADD CONSTRAINT "review_history_old_category_id_fkey" FOREIGN KEY ("old_category_id") REFERENCES "public"."taxonomy_nodes"("id");



ALTER TABLE ONLY "public"."review_history"
    ADD CONSTRAINT "review_history_product_id_fkey" FOREIGN KEY ("product_id") REFERENCES "public"."products"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."similarity_matches"
    ADD CONSTRAINT "similarity_matches_product_a_id_fkey" FOREIGN KEY ("product_a_id") REFERENCES "public"."products"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."similarity_matches"
    ADD CONSTRAINT "similarity_matches_product_b_id_fkey" FOREIGN KEY ("product_b_id") REFERENCES "public"."products"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."synonym_lemmas"
    ADD CONSTRAINT "synonym_lemmas_category_id_fkey" FOREIGN KEY ("category_id") REFERENCES "public"."taxonomy_nodes"("id") ON DELETE SET NULL;



ALTER TABLE ONLY "public"."synonym_terms"
    ADD CONSTRAINT "synonym_terms_lemma_id_fkey" FOREIGN KEY ("lemma_id") REFERENCES "public"."synonym_lemmas"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."system_settings"
    ADD CONSTRAINT "system_settings_updated_by_fkey" FOREIGN KEY ("updated_by") REFERENCES "auth"."users"("id");



ALTER TABLE ONLY "public"."taxonomy_nodes"
    ADD CONSTRAINT "taxonomy_nodes_parent_id_fkey" FOREIGN KEY ("parent_id") REFERENCES "public"."taxonomy_nodes"("id") ON DELETE CASCADE;



ALTER TABLE "public"."audit_logs" ENABLE ROW LEVEL SECURITY;


CREATE POLICY "audit_logs_insert" ON "public"."audit_logs" FOR INSERT WITH CHECK (true);



CREATE POLICY "audit_logs_read" ON "public"."audit_logs" FOR SELECT USING (("auth"."role"() = 'taxonomy_admin'::"text"));



ALTER TABLE "public"."human_feedback" ENABLE ROW LEVEL SECURITY;


CREATE POLICY "human_feedback_insert" ON "public"."human_feedback" FOR INSERT WITH CHECK (("auth"."uid"() = "reviewer_id"));



CREATE POLICY "human_feedback_read" ON "public"."human_feedback" FOR SELECT USING (("auth"."role"() IS NOT NULL));



CREATE POLICY "human_feedback_update" ON "public"."human_feedback" FOR UPDATE USING (("auth"."uid"() = "reviewer_id"));



ALTER TABLE "public"."imports" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."keyword_rules" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."ml_training_history" ENABLE ROW LEVEL SECURITY;


CREATE POLICY "ml_training_history_delete_service" ON "public"."ml_training_history" FOR DELETE USING (true);



CREATE POLICY "ml_training_history_insert_service" ON "public"."ml_training_history" FOR INSERT WITH CHECK (true);



CREATE POLICY "ml_training_history_read_all" ON "public"."ml_training_history" FOR SELECT USING (true);



ALTER TABLE "public"."product_attributes" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."product_category_suggestions" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."products" ENABLE ROW LEVEL SECURITY;


CREATE POLICY "products_delete" ON "public"."products" FOR DELETE USING (("auth"."role"() = 'taxonomy_admin'::"text"));



CREATE POLICY "products_insert" ON "public"."products" FOR INSERT WITH CHECK (("auth"."role"() = ANY (ARRAY['taxonomy_editor'::"text", 'taxonomy_admin'::"text"])));



CREATE POLICY "products_read" ON "public"."products" FOR SELECT USING (true);



CREATE POLICY "products_update" ON "public"."products" FOR UPDATE USING (("auth"."role"() = ANY (ARRAY['taxonomy_editor'::"text", 'taxonomy_admin'::"text"])));



ALTER TABLE "public"."regex_rules" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."review_history" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."similarity_matches" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."synonym_lemmas" ENABLE ROW LEVEL SECURITY;


CREATE POLICY "synonym_lemmas_delete" ON "public"."synonym_lemmas" FOR DELETE USING (("auth"."role"() = ANY (ARRAY['taxonomy_editor'::"text", 'taxonomy_admin'::"text"])));



CREATE POLICY "synonym_lemmas_insert" ON "public"."synonym_lemmas" FOR INSERT WITH CHECK (("auth"."role"() = ANY (ARRAY['taxonomy_editor'::"text", 'taxonomy_admin'::"text"])));



CREATE POLICY "synonym_lemmas_read" ON "public"."synonym_lemmas" FOR SELECT USING (true);



CREATE POLICY "synonym_lemmas_update" ON "public"."synonym_lemmas" FOR UPDATE USING (("auth"."role"() = ANY (ARRAY['taxonomy_editor'::"text", 'taxonomy_admin'::"text"])));



ALTER TABLE "public"."synonym_terms" ENABLE ROW LEVEL SECURITY;


CREATE POLICY "synonym_terms_delete" ON "public"."synonym_terms" FOR DELETE USING (("auth"."role"() = ANY (ARRAY['taxonomy_editor'::"text", 'taxonomy_admin'::"text"])));



CREATE POLICY "synonym_terms_insert" ON "public"."synonym_terms" FOR INSERT WITH CHECK (("auth"."role"() = ANY (ARRAY['taxonomy_editor'::"text", 'taxonomy_admin'::"text"])));



CREATE POLICY "synonym_terms_read" ON "public"."synonym_terms" FOR SELECT USING (true);



CREATE POLICY "synonym_terms_update" ON "public"."synonym_terms" FOR UPDATE USING (("auth"."role"() = ANY (ARRAY['taxonomy_editor'::"text", 'taxonomy_admin'::"text"])));



ALTER TABLE "public"."taxonomy_nodes" ENABLE ROW LEVEL SECURITY;


CREATE POLICY "taxonomy_nodes_delete" ON "public"."taxonomy_nodes" FOR DELETE USING (("auth"."role"() = 'taxonomy_admin'::"text"));



CREATE POLICY "taxonomy_nodes_insert" ON "public"."taxonomy_nodes" FOR INSERT WITH CHECK (("auth"."role"() = ANY (ARRAY['taxonomy_editor'::"text", 'taxonomy_admin'::"text"])));



CREATE POLICY "taxonomy_nodes_read" ON "public"."taxonomy_nodes" FOR SELECT USING (true);



CREATE POLICY "taxonomy_nodes_update" ON "public"."taxonomy_nodes" FOR UPDATE USING (("auth"."role"() = ANY (ARRAY['taxonomy_editor'::"text", 'taxonomy_admin'::"text"])));





ALTER PUBLICATION "supabase_realtime" OWNER TO "postgres";





GRANT USAGE ON SCHEMA "public" TO "postgres";
GRANT USAGE ON SCHEMA "public" TO "anon";
GRANT USAGE ON SCHEMA "public" TO "authenticated";
GRANT USAGE ON SCHEMA "public" TO "service_role";
GRANT USAGE ON SCHEMA "public" TO "taxonomy_reader";



GRANT ALL ON FUNCTION "public"."halfvec_in"("cstring", "oid", integer) TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_in"("cstring", "oid", integer) TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_in"("cstring", "oid", integer) TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_in"("cstring", "oid", integer) TO "service_role";



GRANT ALL ON FUNCTION "public"."halfvec_out"("public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_out"("public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_out"("public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_out"("public"."halfvec") TO "service_role";



GRANT ALL ON FUNCTION "public"."halfvec_recv"("internal", "oid", integer) TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_recv"("internal", "oid", integer) TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_recv"("internal", "oid", integer) TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_recv"("internal", "oid", integer) TO "service_role";



GRANT ALL ON FUNCTION "public"."halfvec_send"("public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_send"("public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_send"("public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_send"("public"."halfvec") TO "service_role";



GRANT ALL ON FUNCTION "public"."halfvec_typmod_in"("cstring"[]) TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_typmod_in"("cstring"[]) TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_typmod_in"("cstring"[]) TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_typmod_in"("cstring"[]) TO "service_role";



GRANT ALL ON FUNCTION "public"."sparsevec_in"("cstring", "oid", integer) TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_in"("cstring", "oid", integer) TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_in"("cstring", "oid", integer) TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_in"("cstring", "oid", integer) TO "service_role";



GRANT ALL ON FUNCTION "public"."sparsevec_out"("public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_out"("public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_out"("public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_out"("public"."sparsevec") TO "service_role";



GRANT ALL ON FUNCTION "public"."sparsevec_recv"("internal", "oid", integer) TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_recv"("internal", "oid", integer) TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_recv"("internal", "oid", integer) TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_recv"("internal", "oid", integer) TO "service_role";



GRANT ALL ON FUNCTION "public"."sparsevec_send"("public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_send"("public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_send"("public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_send"("public"."sparsevec") TO "service_role";



GRANT ALL ON FUNCTION "public"."sparsevec_typmod_in"("cstring"[]) TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_typmod_in"("cstring"[]) TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_typmod_in"("cstring"[]) TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_typmod_in"("cstring"[]) TO "service_role";



GRANT ALL ON FUNCTION "public"."vector_in"("cstring", "oid", integer) TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_in"("cstring", "oid", integer) TO "anon";
GRANT ALL ON FUNCTION "public"."vector_in"("cstring", "oid", integer) TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_in"("cstring", "oid", integer) TO "service_role";



GRANT ALL ON FUNCTION "public"."vector_out"("public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_out"("public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_out"("public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_out"("public"."vector") TO "service_role";



GRANT ALL ON FUNCTION "public"."vector_recv"("internal", "oid", integer) TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_recv"("internal", "oid", integer) TO "anon";
GRANT ALL ON FUNCTION "public"."vector_recv"("internal", "oid", integer) TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_recv"("internal", "oid", integer) TO "service_role";



GRANT ALL ON FUNCTION "public"."vector_send"("public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_send"("public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_send"("public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_send"("public"."vector") TO "service_role";



GRANT ALL ON FUNCTION "public"."vector_typmod_in"("cstring"[]) TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_typmod_in"("cstring"[]) TO "anon";
GRANT ALL ON FUNCTION "public"."vector_typmod_in"("cstring"[]) TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_typmod_in"("cstring"[]) TO "service_role";



GRANT ALL ON FUNCTION "public"."array_to_halfvec"(real[], integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."array_to_halfvec"(real[], integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."array_to_halfvec"(real[], integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."array_to_halfvec"(real[], integer, boolean) TO "service_role";



GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(real[], integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(real[], integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(real[], integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(real[], integer, boolean) TO "service_role";



GRANT ALL ON FUNCTION "public"."array_to_vector"(real[], integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."array_to_vector"(real[], integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."array_to_vector"(real[], integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."array_to_vector"(real[], integer, boolean) TO "service_role";



GRANT ALL ON FUNCTION "public"."array_to_halfvec"(double precision[], integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."array_to_halfvec"(double precision[], integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."array_to_halfvec"(double precision[], integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."array_to_halfvec"(double precision[], integer, boolean) TO "service_role";



GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(double precision[], integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(double precision[], integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(double precision[], integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(double precision[], integer, boolean) TO "service_role";



GRANT ALL ON FUNCTION "public"."array_to_vector"(double precision[], integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."array_to_vector"(double precision[], integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."array_to_vector"(double precision[], integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."array_to_vector"(double precision[], integer, boolean) TO "service_role";



GRANT ALL ON FUNCTION "public"."array_to_halfvec"(integer[], integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."array_to_halfvec"(integer[], integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."array_to_halfvec"(integer[], integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."array_to_halfvec"(integer[], integer, boolean) TO "service_role";



GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(integer[], integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(integer[], integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(integer[], integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(integer[], integer, boolean) TO "service_role";



GRANT ALL ON FUNCTION "public"."array_to_vector"(integer[], integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."array_to_vector"(integer[], integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."array_to_vector"(integer[], integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."array_to_vector"(integer[], integer, boolean) TO "service_role";



GRANT ALL ON FUNCTION "public"."array_to_halfvec"(numeric[], integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."array_to_halfvec"(numeric[], integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."array_to_halfvec"(numeric[], integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."array_to_halfvec"(numeric[], integer, boolean) TO "service_role";



GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(numeric[], integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(numeric[], integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(numeric[], integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(numeric[], integer, boolean) TO "service_role";



GRANT ALL ON FUNCTION "public"."array_to_vector"(numeric[], integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."array_to_vector"(numeric[], integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."array_to_vector"(numeric[], integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."array_to_vector"(numeric[], integer, boolean) TO "service_role";



GRANT ALL ON FUNCTION "public"."halfvec_to_float4"("public"."halfvec", integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_to_float4"("public"."halfvec", integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_to_float4"("public"."halfvec", integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_to_float4"("public"."halfvec", integer, boolean) TO "service_role";



GRANT ALL ON FUNCTION "public"."halfvec"("public"."halfvec", integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec"("public"."halfvec", integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec"("public"."halfvec", integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec"("public"."halfvec", integer, boolean) TO "service_role";



GRANT ALL ON FUNCTION "public"."halfvec_to_sparsevec"("public"."halfvec", integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_to_sparsevec"("public"."halfvec", integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_to_sparsevec"("public"."halfvec", integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_to_sparsevec"("public"."halfvec", integer, boolean) TO "service_role";



GRANT ALL ON FUNCTION "public"."halfvec_to_vector"("public"."halfvec", integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_to_vector"("public"."halfvec", integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_to_vector"("public"."halfvec", integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_to_vector"("public"."halfvec", integer, boolean) TO "service_role";



GRANT ALL ON FUNCTION "public"."sparsevec_to_halfvec"("public"."sparsevec", integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_to_halfvec"("public"."sparsevec", integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_to_halfvec"("public"."sparsevec", integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_to_halfvec"("public"."sparsevec", integer, boolean) TO "service_role";



GRANT ALL ON FUNCTION "public"."sparsevec"("public"."sparsevec", integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec"("public"."sparsevec", integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec"("public"."sparsevec", integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec"("public"."sparsevec", integer, boolean) TO "service_role";



GRANT ALL ON FUNCTION "public"."sparsevec_to_vector"("public"."sparsevec", integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_to_vector"("public"."sparsevec", integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_to_vector"("public"."sparsevec", integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_to_vector"("public"."sparsevec", integer, boolean) TO "service_role";



GRANT ALL ON FUNCTION "public"."vector_to_float4"("public"."vector", integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_to_float4"("public"."vector", integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."vector_to_float4"("public"."vector", integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_to_float4"("public"."vector", integer, boolean) TO "service_role";



GRANT ALL ON FUNCTION "public"."vector_to_halfvec"("public"."vector", integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_to_halfvec"("public"."vector", integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."vector_to_halfvec"("public"."vector", integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_to_halfvec"("public"."vector", integer, boolean) TO "service_role";



GRANT ALL ON FUNCTION "public"."vector_to_sparsevec"("public"."vector", integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_to_sparsevec"("public"."vector", integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."vector_to_sparsevec"("public"."vector", integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_to_sparsevec"("public"."vector", integer, boolean) TO "service_role";



GRANT ALL ON FUNCTION "public"."vector"("public"."vector", integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."vector"("public"."vector", integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."vector"("public"."vector", integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector"("public"."vector", integer, boolean) TO "service_role";




























































































































































GRANT ALL ON FUNCTION "public"."batch_category_classification"("product_data" "jsonb", "top_k" integer) TO "anon";
GRANT ALL ON FUNCTION "public"."batch_category_classification"("product_data" "jsonb", "top_k" integer) TO "authenticated";



GRANT ALL ON FUNCTION "public"."binary_quantize"("public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."binary_quantize"("public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."binary_quantize"("public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."binary_quantize"("public"."halfvec") TO "service_role";



GRANT ALL ON FUNCTION "public"."binary_quantize"("public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."binary_quantize"("public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."binary_quantize"("public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."binary_quantize"("public"."vector") TO "service_role";



GRANT ALL ON FUNCTION "public"."cosine_distance"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."cosine_distance"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."cosine_distance"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."cosine_distance"("public"."halfvec", "public"."halfvec") TO "service_role";



GRANT ALL ON FUNCTION "public"."cosine_distance"("public"."sparsevec", "public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."cosine_distance"("public"."sparsevec", "public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."cosine_distance"("public"."sparsevec", "public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."cosine_distance"("public"."sparsevec", "public"."sparsevec") TO "service_role";



GRANT ALL ON FUNCTION "public"."cosine_distance"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."cosine_distance"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."cosine_distance"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."cosine_distance"("public"."vector", "public"."vector") TO "service_role";



GRANT ALL ON FUNCTION "public"."exec_sql"("query_text" "text", "query_params" "jsonb") TO "service_role";



GRANT ALL ON FUNCTION "public"."get_sample_categories_with_embeddings"("sample_size" integer) TO "anon";
GRANT ALL ON FUNCTION "public"."get_sample_categories_with_embeddings"("sample_size" integer) TO "authenticated";



GRANT ALL ON FUNCTION "public"."halfvec_accum"(double precision[], "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_accum"(double precision[], "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_accum"(double precision[], "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_accum"(double precision[], "public"."halfvec") TO "service_role";



GRANT ALL ON FUNCTION "public"."halfvec_add"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_add"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_add"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_add"("public"."halfvec", "public"."halfvec") TO "service_role";



GRANT ALL ON FUNCTION "public"."halfvec_avg"(double precision[]) TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_avg"(double precision[]) TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_avg"(double precision[]) TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_avg"(double precision[]) TO "service_role";



GRANT ALL ON FUNCTION "public"."halfvec_cmp"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_cmp"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_cmp"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_cmp"("public"."halfvec", "public"."halfvec") TO "service_role";



GRANT ALL ON FUNCTION "public"."halfvec_combine"(double precision[], double precision[]) TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_combine"(double precision[], double precision[]) TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_combine"(double precision[], double precision[]) TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_combine"(double precision[], double precision[]) TO "service_role";



GRANT ALL ON FUNCTION "public"."halfvec_concat"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_concat"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_concat"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_concat"("public"."halfvec", "public"."halfvec") TO "service_role";



GRANT ALL ON FUNCTION "public"."halfvec_eq"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_eq"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_eq"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_eq"("public"."halfvec", "public"."halfvec") TO "service_role";



GRANT ALL ON FUNCTION "public"."halfvec_ge"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_ge"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_ge"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_ge"("public"."halfvec", "public"."halfvec") TO "service_role";



GRANT ALL ON FUNCTION "public"."halfvec_gt"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_gt"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_gt"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_gt"("public"."halfvec", "public"."halfvec") TO "service_role";



GRANT ALL ON FUNCTION "public"."halfvec_l2_squared_distance"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_l2_squared_distance"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_l2_squared_distance"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_l2_squared_distance"("public"."halfvec", "public"."halfvec") TO "service_role";



GRANT ALL ON FUNCTION "public"."halfvec_le"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_le"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_le"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_le"("public"."halfvec", "public"."halfvec") TO "service_role";



GRANT ALL ON FUNCTION "public"."halfvec_lt"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_lt"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_lt"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_lt"("public"."halfvec", "public"."halfvec") TO "service_role";



GRANT ALL ON FUNCTION "public"."halfvec_mul"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_mul"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_mul"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_mul"("public"."halfvec", "public"."halfvec") TO "service_role";



GRANT ALL ON FUNCTION "public"."halfvec_ne"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_ne"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_ne"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_ne"("public"."halfvec", "public"."halfvec") TO "service_role";



GRANT ALL ON FUNCTION "public"."halfvec_negative_inner_product"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_negative_inner_product"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_negative_inner_product"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_negative_inner_product"("public"."halfvec", "public"."halfvec") TO "service_role";



GRANT ALL ON FUNCTION "public"."halfvec_spherical_distance"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_spherical_distance"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_spherical_distance"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_spherical_distance"("public"."halfvec", "public"."halfvec") TO "service_role";



GRANT ALL ON FUNCTION "public"."halfvec_sub"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_sub"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_sub"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_sub"("public"."halfvec", "public"."halfvec") TO "service_role";



GRANT ALL ON FUNCTION "public"."hamming_distance"(bit, bit) TO "postgres";
GRANT ALL ON FUNCTION "public"."hamming_distance"(bit, bit) TO "anon";
GRANT ALL ON FUNCTION "public"."hamming_distance"(bit, bit) TO "authenticated";
GRANT ALL ON FUNCTION "public"."hamming_distance"(bit, bit) TO "service_role";



GRANT ALL ON FUNCTION "public"."hnsw_bit_support"("internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."hnsw_bit_support"("internal") TO "anon";
GRANT ALL ON FUNCTION "public"."hnsw_bit_support"("internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."hnsw_bit_support"("internal") TO "service_role";



GRANT ALL ON FUNCTION "public"."hnsw_halfvec_support"("internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."hnsw_halfvec_support"("internal") TO "anon";
GRANT ALL ON FUNCTION "public"."hnsw_halfvec_support"("internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."hnsw_halfvec_support"("internal") TO "service_role";



GRANT ALL ON FUNCTION "public"."hnsw_sparsevec_support"("internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."hnsw_sparsevec_support"("internal") TO "anon";
GRANT ALL ON FUNCTION "public"."hnsw_sparsevec_support"("internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."hnsw_sparsevec_support"("internal") TO "service_role";



GRANT ALL ON FUNCTION "public"."hnswhandler"("internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."hnswhandler"("internal") TO "anon";
GRANT ALL ON FUNCTION "public"."hnswhandler"("internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."hnswhandler"("internal") TO "service_role";



GRANT ALL ON FUNCTION "public"."hybrid_category_classification"("product_name" "text", "product_embedding" "public"."vector", "top_k" integer) TO "anon";
GRANT ALL ON FUNCTION "public"."hybrid_category_classification"("product_name" "text", "product_embedding" "public"."vector", "top_k" integer) TO "authenticated";



GRANT ALL ON FUNCTION "public"."inner_product"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."inner_product"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."inner_product"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."inner_product"("public"."halfvec", "public"."halfvec") TO "service_role";



GRANT ALL ON FUNCTION "public"."inner_product"("public"."sparsevec", "public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."inner_product"("public"."sparsevec", "public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."inner_product"("public"."sparsevec", "public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."inner_product"("public"."sparsevec", "public"."sparsevec") TO "service_role";



GRANT ALL ON FUNCTION "public"."inner_product"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."inner_product"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."inner_product"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."inner_product"("public"."vector", "public"."vector") TO "service_role";



GRANT ALL ON FUNCTION "public"."ivfflat_bit_support"("internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."ivfflat_bit_support"("internal") TO "anon";
GRANT ALL ON FUNCTION "public"."ivfflat_bit_support"("internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."ivfflat_bit_support"("internal") TO "service_role";



GRANT ALL ON FUNCTION "public"."ivfflat_halfvec_support"("internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."ivfflat_halfvec_support"("internal") TO "anon";
GRANT ALL ON FUNCTION "public"."ivfflat_halfvec_support"("internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."ivfflat_halfvec_support"("internal") TO "service_role";



GRANT ALL ON FUNCTION "public"."ivfflathandler"("internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."ivfflathandler"("internal") TO "anon";
GRANT ALL ON FUNCTION "public"."ivfflathandler"("internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."ivfflathandler"("internal") TO "service_role";



GRANT ALL ON FUNCTION "public"."jaccard_distance"(bit, bit) TO "postgres";
GRANT ALL ON FUNCTION "public"."jaccard_distance"(bit, bit) TO "anon";
GRANT ALL ON FUNCTION "public"."jaccard_distance"(bit, bit) TO "authenticated";
GRANT ALL ON FUNCTION "public"."jaccard_distance"(bit, bit) TO "service_role";



GRANT ALL ON FUNCTION "public"."l1_distance"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."l1_distance"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."l1_distance"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."l1_distance"("public"."halfvec", "public"."halfvec") TO "service_role";



GRANT ALL ON FUNCTION "public"."l1_distance"("public"."sparsevec", "public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."l1_distance"("public"."sparsevec", "public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."l1_distance"("public"."sparsevec", "public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."l1_distance"("public"."sparsevec", "public"."sparsevec") TO "service_role";



GRANT ALL ON FUNCTION "public"."l1_distance"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."l1_distance"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."l1_distance"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."l1_distance"("public"."vector", "public"."vector") TO "service_role";



GRANT ALL ON FUNCTION "public"."l2_distance"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."l2_distance"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."l2_distance"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."l2_distance"("public"."halfvec", "public"."halfvec") TO "service_role";



GRANT ALL ON FUNCTION "public"."l2_distance"("public"."sparsevec", "public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."l2_distance"("public"."sparsevec", "public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."l2_distance"("public"."sparsevec", "public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."l2_distance"("public"."sparsevec", "public"."sparsevec") TO "service_role";



GRANT ALL ON FUNCTION "public"."l2_distance"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."l2_distance"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."l2_distance"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."l2_distance"("public"."vector", "public"."vector") TO "service_role";



GRANT ALL ON FUNCTION "public"."l2_norm"("public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."l2_norm"("public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."l2_norm"("public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."l2_norm"("public"."halfvec") TO "service_role";



GRANT ALL ON FUNCTION "public"."l2_norm"("public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."l2_norm"("public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."l2_norm"("public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."l2_norm"("public"."sparsevec") TO "service_role";



GRANT ALL ON FUNCTION "public"."l2_normalize"("public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."l2_normalize"("public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."l2_normalize"("public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."l2_normalize"("public"."halfvec") TO "service_role";



GRANT ALL ON FUNCTION "public"."l2_normalize"("public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."l2_normalize"("public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."l2_normalize"("public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."l2_normalize"("public"."sparsevec") TO "service_role";



GRANT ALL ON FUNCTION "public"."l2_normalize"("public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."l2_normalize"("public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."l2_normalize"("public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."l2_normalize"("public"."vector") TO "service_role";



GRANT ALL ON FUNCTION "public"."match_categories_by_embedding"("query_embedding" "public"."vector", "match_threshold" double precision, "match_count" integer) TO "anon";
GRANT ALL ON FUNCTION "public"."match_categories_by_embedding"("query_embedding" "public"."vector", "match_threshold" double precision, "match_count" integer) TO "authenticated";



GRANT ALL ON FUNCTION "public"."sparsevec_cmp"("public"."sparsevec", "public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_cmp"("public"."sparsevec", "public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_cmp"("public"."sparsevec", "public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_cmp"("public"."sparsevec", "public"."sparsevec") TO "service_role";



GRANT ALL ON FUNCTION "public"."sparsevec_eq"("public"."sparsevec", "public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_eq"("public"."sparsevec", "public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_eq"("public"."sparsevec", "public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_eq"("public"."sparsevec", "public"."sparsevec") TO "service_role";



GRANT ALL ON FUNCTION "public"."sparsevec_ge"("public"."sparsevec", "public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_ge"("public"."sparsevec", "public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_ge"("public"."sparsevec", "public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_ge"("public"."sparsevec", "public"."sparsevec") TO "service_role";



GRANT ALL ON FUNCTION "public"."sparsevec_gt"("public"."sparsevec", "public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_gt"("public"."sparsevec", "public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_gt"("public"."sparsevec", "public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_gt"("public"."sparsevec", "public"."sparsevec") TO "service_role";



GRANT ALL ON FUNCTION "public"."sparsevec_l2_squared_distance"("public"."sparsevec", "public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_l2_squared_distance"("public"."sparsevec", "public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_l2_squared_distance"("public"."sparsevec", "public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_l2_squared_distance"("public"."sparsevec", "public"."sparsevec") TO "service_role";



GRANT ALL ON FUNCTION "public"."sparsevec_le"("public"."sparsevec", "public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_le"("public"."sparsevec", "public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_le"("public"."sparsevec", "public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_le"("public"."sparsevec", "public"."sparsevec") TO "service_role";



GRANT ALL ON FUNCTION "public"."sparsevec_lt"("public"."sparsevec", "public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_lt"("public"."sparsevec", "public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_lt"("public"."sparsevec", "public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_lt"("public"."sparsevec", "public"."sparsevec") TO "service_role";



GRANT ALL ON FUNCTION "public"."sparsevec_ne"("public"."sparsevec", "public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_ne"("public"."sparsevec", "public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_ne"("public"."sparsevec", "public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_ne"("public"."sparsevec", "public"."sparsevec") TO "service_role";



GRANT ALL ON FUNCTION "public"."sparsevec_negative_inner_product"("public"."sparsevec", "public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_negative_inner_product"("public"."sparsevec", "public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_negative_inner_product"("public"."sparsevec", "public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_negative_inner_product"("public"."sparsevec", "public"."sparsevec") TO "service_role";



GRANT ALL ON FUNCTION "public"."subvector"("public"."halfvec", integer, integer) TO "postgres";
GRANT ALL ON FUNCTION "public"."subvector"("public"."halfvec", integer, integer) TO "anon";
GRANT ALL ON FUNCTION "public"."subvector"("public"."halfvec", integer, integer) TO "authenticated";
GRANT ALL ON FUNCTION "public"."subvector"("public"."halfvec", integer, integer) TO "service_role";



GRANT ALL ON FUNCTION "public"."subvector"("public"."vector", integer, integer) TO "postgres";
GRANT ALL ON FUNCTION "public"."subvector"("public"."vector", integer, integer) TO "anon";
GRANT ALL ON FUNCTION "public"."subvector"("public"."vector", integer, integer) TO "authenticated";
GRANT ALL ON FUNCTION "public"."subvector"("public"."vector", integer, integer) TO "service_role";



GRANT ALL ON FUNCTION "public"."vector_accum"(double precision[], "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_accum"(double precision[], "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_accum"(double precision[], "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_accum"(double precision[], "public"."vector") TO "service_role";



GRANT ALL ON FUNCTION "public"."vector_add"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_add"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_add"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_add"("public"."vector", "public"."vector") TO "service_role";



GRANT ALL ON FUNCTION "public"."vector_avg"(double precision[]) TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_avg"(double precision[]) TO "anon";
GRANT ALL ON FUNCTION "public"."vector_avg"(double precision[]) TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_avg"(double precision[]) TO "service_role";



GRANT ALL ON FUNCTION "public"."vector_cmp"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_cmp"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_cmp"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_cmp"("public"."vector", "public"."vector") TO "service_role";



GRANT ALL ON FUNCTION "public"."vector_combine"(double precision[], double precision[]) TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_combine"(double precision[], double precision[]) TO "anon";
GRANT ALL ON FUNCTION "public"."vector_combine"(double precision[], double precision[]) TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_combine"(double precision[], double precision[]) TO "service_role";



GRANT ALL ON FUNCTION "public"."vector_concat"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_concat"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_concat"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_concat"("public"."vector", "public"."vector") TO "service_role";



GRANT ALL ON FUNCTION "public"."vector_dims"("public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_dims"("public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_dims"("public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_dims"("public"."halfvec") TO "service_role";



GRANT ALL ON FUNCTION "public"."vector_dims"("public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_dims"("public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_dims"("public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_dims"("public"."vector") TO "service_role";



GRANT ALL ON FUNCTION "public"."vector_eq"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_eq"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_eq"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_eq"("public"."vector", "public"."vector") TO "service_role";



GRANT ALL ON FUNCTION "public"."vector_ge"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_ge"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_ge"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_ge"("public"."vector", "public"."vector") TO "service_role";



GRANT ALL ON FUNCTION "public"."vector_gt"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_gt"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_gt"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_gt"("public"."vector", "public"."vector") TO "service_role";



GRANT ALL ON FUNCTION "public"."vector_l2_squared_distance"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_l2_squared_distance"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_l2_squared_distance"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_l2_squared_distance"("public"."vector", "public"."vector") TO "service_role";



GRANT ALL ON FUNCTION "public"."vector_le"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_le"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_le"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_le"("public"."vector", "public"."vector") TO "service_role";



GRANT ALL ON FUNCTION "public"."vector_lt"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_lt"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_lt"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_lt"("public"."vector", "public"."vector") TO "service_role";



GRANT ALL ON FUNCTION "public"."vector_mul"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_mul"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_mul"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_mul"("public"."vector", "public"."vector") TO "service_role";



GRANT ALL ON FUNCTION "public"."vector_ne"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_ne"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_ne"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_ne"("public"."vector", "public"."vector") TO "service_role";



GRANT ALL ON FUNCTION "public"."vector_negative_inner_product"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_negative_inner_product"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_negative_inner_product"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_negative_inner_product"("public"."vector", "public"."vector") TO "service_role";



GRANT ALL ON FUNCTION "public"."vector_norm"("public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_norm"("public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_norm"("public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_norm"("public"."vector") TO "service_role";



GRANT ALL ON FUNCTION "public"."vector_spherical_distance"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_spherical_distance"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_spherical_distance"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_spherical_distance"("public"."vector", "public"."vector") TO "service_role";



GRANT ALL ON FUNCTION "public"."vector_sub"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_sub"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_sub"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_sub"("public"."vector", "public"."vector") TO "service_role";












GRANT ALL ON FUNCTION "public"."avg"("public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."avg"("public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."avg"("public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."avg"("public"."halfvec") TO "service_role";



GRANT ALL ON FUNCTION "public"."avg"("public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."avg"("public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."avg"("public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."avg"("public"."vector") TO "service_role";



GRANT ALL ON FUNCTION "public"."sum"("public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."sum"("public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."sum"("public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."sum"("public"."halfvec") TO "service_role";



GRANT ALL ON FUNCTION "public"."sum"("public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."sum"("public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."sum"("public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."sum"("public"."vector") TO "service_role";









GRANT ALL ON TABLE "public"."audit_logs" TO "anon";
GRANT ALL ON TABLE "public"."audit_logs" TO "authenticated";
GRANT ALL ON TABLE "public"."audit_logs" TO "service_role";
GRANT SELECT ON TABLE "public"."audit_logs" TO "taxonomy_reader";
GRANT ALL ON TABLE "public"."audit_logs" TO "taxonomy_admin";



GRANT ALL ON TABLE "public"."human_feedback" TO "anon";
GRANT ALL ON TABLE "public"."human_feedback" TO "authenticated";
GRANT ALL ON TABLE "public"."human_feedback" TO "service_role";



GRANT ALL ON TABLE "public"."imports" TO "anon";
GRANT ALL ON TABLE "public"."imports" TO "authenticated";
GRANT ALL ON TABLE "public"."imports" TO "service_role";
GRANT SELECT ON TABLE "public"."imports" TO "taxonomy_reader";
GRANT INSERT,UPDATE ON TABLE "public"."imports" TO "taxonomy_editor";
GRANT ALL ON TABLE "public"."imports" TO "taxonomy_admin";



GRANT ALL ON TABLE "public"."keyword_rules" TO "anon";
GRANT ALL ON TABLE "public"."keyword_rules" TO "authenticated";
GRANT ALL ON TABLE "public"."keyword_rules" TO "service_role";
GRANT SELECT ON TABLE "public"."keyword_rules" TO "taxonomy_reader";
GRANT INSERT,DELETE,UPDATE ON TABLE "public"."keyword_rules" TO "taxonomy_editor";
GRANT ALL ON TABLE "public"."keyword_rules" TO "taxonomy_admin";



GRANT ALL ON TABLE "public"."ml_training_history" TO "anon";
GRANT ALL ON TABLE "public"."ml_training_history" TO "authenticated";
GRANT ALL ON TABLE "public"."ml_training_history" TO "service_role";



GRANT ALL ON TABLE "public"."product_attributes" TO "anon";
GRANT ALL ON TABLE "public"."product_attributes" TO "authenticated";
GRANT ALL ON TABLE "public"."product_attributes" TO "service_role";
GRANT SELECT ON TABLE "public"."product_attributes" TO "taxonomy_reader";
GRANT INSERT,DELETE,UPDATE ON TABLE "public"."product_attributes" TO "taxonomy_editor";
GRANT ALL ON TABLE "public"."product_attributes" TO "taxonomy_admin";



GRANT ALL ON TABLE "public"."product_category_suggestions" TO "anon";
GRANT ALL ON TABLE "public"."product_category_suggestions" TO "authenticated";
GRANT ALL ON TABLE "public"."product_category_suggestions" TO "service_role";
GRANT SELECT ON TABLE "public"."product_category_suggestions" TO "taxonomy_reader";
GRANT INSERT,DELETE,UPDATE ON TABLE "public"."product_category_suggestions" TO "taxonomy_editor";
GRANT ALL ON TABLE "public"."product_category_suggestions" TO "taxonomy_admin";



GRANT ALL ON TABLE "public"."products" TO "anon";
GRANT ALL ON TABLE "public"."products" TO "authenticated";
GRANT ALL ON TABLE "public"."products" TO "service_role";
GRANT SELECT ON TABLE "public"."products" TO "taxonomy_reader";
GRANT INSERT,DELETE,UPDATE ON TABLE "public"."products" TO "taxonomy_editor";
GRANT ALL ON TABLE "public"."products" TO "taxonomy_admin";



GRANT ALL ON TABLE "public"."regex_rules" TO "anon";
GRANT ALL ON TABLE "public"."regex_rules" TO "authenticated";
GRANT ALL ON TABLE "public"."regex_rules" TO "service_role";
GRANT SELECT ON TABLE "public"."regex_rules" TO "taxonomy_reader";
GRANT INSERT,DELETE,UPDATE ON TABLE "public"."regex_rules" TO "taxonomy_editor";
GRANT ALL ON TABLE "public"."regex_rules" TO "taxonomy_admin";



GRANT ALL ON TABLE "public"."review_history" TO "anon";
GRANT ALL ON TABLE "public"."review_history" TO "authenticated";
GRANT ALL ON TABLE "public"."review_history" TO "service_role";
GRANT SELECT ON TABLE "public"."review_history" TO "taxonomy_reader";
GRANT INSERT,DELETE,UPDATE ON TABLE "public"."review_history" TO "taxonomy_editor";
GRANT ALL ON TABLE "public"."review_history" TO "taxonomy_admin";



GRANT ALL ON TABLE "public"."similarity_matches" TO "anon";
GRANT ALL ON TABLE "public"."similarity_matches" TO "authenticated";
GRANT ALL ON TABLE "public"."similarity_matches" TO "service_role";
GRANT SELECT ON TABLE "public"."similarity_matches" TO "taxonomy_reader";
GRANT INSERT,DELETE,UPDATE ON TABLE "public"."similarity_matches" TO "taxonomy_editor";
GRANT ALL ON TABLE "public"."similarity_matches" TO "taxonomy_admin";



GRANT ALL ON TABLE "public"."synonym_lemmas" TO "anon";
GRANT ALL ON TABLE "public"."synonym_lemmas" TO "authenticated";
GRANT ALL ON TABLE "public"."synonym_lemmas" TO "service_role";
GRANT SELECT ON TABLE "public"."synonym_lemmas" TO "taxonomy_reader";
GRANT INSERT,DELETE,UPDATE ON TABLE "public"."synonym_lemmas" TO "taxonomy_editor";
GRANT ALL ON TABLE "public"."synonym_lemmas" TO "taxonomy_admin";



GRANT ALL ON TABLE "public"."synonym_terms" TO "anon";
GRANT ALL ON TABLE "public"."synonym_terms" TO "authenticated";
GRANT ALL ON TABLE "public"."synonym_terms" TO "service_role";
GRANT SELECT ON TABLE "public"."synonym_terms" TO "taxonomy_reader";
GRANT INSERT,DELETE,UPDATE ON TABLE "public"."synonym_terms" TO "taxonomy_editor";
GRANT ALL ON TABLE "public"."synonym_terms" TO "taxonomy_admin";



GRANT ALL ON TABLE "public"."system_settings" TO "anon";
GRANT ALL ON TABLE "public"."system_settings" TO "authenticated";
GRANT ALL ON TABLE "public"."system_settings" TO "service_role";
GRANT SELECT ON TABLE "public"."system_settings" TO "taxonomy_reader";
GRANT ALL ON TABLE "public"."system_settings" TO "taxonomy_admin";



GRANT ALL ON TABLE "public"."taxonomy_nodes" TO "anon";
GRANT ALL ON TABLE "public"."taxonomy_nodes" TO "authenticated";
GRANT ALL ON TABLE "public"."taxonomy_nodes" TO "service_role";
GRANT SELECT ON TABLE "public"."taxonomy_nodes" TO "taxonomy_reader";
GRANT INSERT,DELETE,UPDATE ON TABLE "public"."taxonomy_nodes" TO "taxonomy_editor";
GRANT ALL ON TABLE "public"."taxonomy_nodes" TO "taxonomy_admin";









ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT UPDATE ON SEQUENCES TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT UPDATE ON SEQUENCES TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT UPDATE ON SEQUENCES TO "service_role";






ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS TO "postgres";






ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLES TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLES TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT REFERENCES,TRIGGER,TRUNCATE,MAINTAIN ON TABLES TO "service_role";































