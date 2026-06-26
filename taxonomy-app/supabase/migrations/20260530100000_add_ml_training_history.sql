-- ============================================================
-- Migration: ML Training History Table
-- Created: 2026-05-30
-- Purpose: เก็บประวัติการเทรนโมเดล ML ถาวรใน Supabase
--          แทนการเก็บใน memory (ซึ่งหายเมื่อ Restart Server)
-- ============================================================

CREATE TABLE IF NOT EXISTS ml_training_history (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    
    -- เมตาข้อมูลการเทรน
    training_date TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    model_type TEXT NOT NULL DEFAULT 'random_forest',
    
    -- ขนาดข้อมูล
    total_samples INTEGER NOT NULL DEFAULT 0,
    train_samples INTEGER NOT NULL DEFAULT 0,
    test_samples  INTEGER NOT NULL DEFAULT 0,
    
    -- ผลลัพธ์ความแม่นยำ
    train_accuracy FLOAT NOT NULL DEFAULT 0,
    test_accuracy  FLOAT NOT NULL DEFAULT 0,
    cv_mean_accuracy FLOAT DEFAULT 0,
    cv_std_accuracy  FLOAT DEFAULT 0,
    
    -- ข้อมูลโมเดลเชิงลึก (JSON)
    feature_importance JSONB DEFAULT '[]'::jsonb,
    classes            JSONB DEFAULT '[]'::jsonb,
    classification_report JSONB DEFAULT '{}'::jsonb,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index สำหรับ query ล่าสุดก่อน
CREATE INDEX IF NOT EXISTS idx_ml_training_history_date 
    ON ml_training_history(training_date DESC);

CREATE INDEX IF NOT EXISTS idx_ml_training_history_accuracy
    ON ml_training_history(test_accuracy DESC);

-- ========================================
-- RLS Policies
-- ========================================
ALTER TABLE ml_training_history ENABLE ROW LEVEL SECURITY;

-- อ่านได้สาธารณะ (anon + authenticated)
CREATE POLICY "ml_training_history_read_all"
    ON ml_training_history FOR SELECT
    USING (true);

-- เขียนได้เฉพาะ service_role (FastAPI Backend)
CREATE POLICY "ml_training_history_insert_service"
    ON ml_training_history FOR INSERT
    WITH CHECK (true);

-- ลบได้เฉพาะ service_role
CREATE POLICY "ml_training_history_delete_service"
    ON ml_training_history FOR DELETE
    USING (true);

-- ========================================
-- Comments
-- ========================================
COMMENT ON TABLE ml_training_history IS 
    'เก็บประวัติการเทรนโมเดล ML ทุก session — สร้างโดย Phayak 2026-05-30';

COMMENT ON COLUMN ml_training_history.feature_importance IS
    'Array ของ {feature: string, importance: float} เรียงตามความสำคัญ';

COMMENT ON COLUMN ml_training_history.classes IS
    'รายการ class labels ที่โมเดลรู้จัก เช่น ["duplicate","different"]';
