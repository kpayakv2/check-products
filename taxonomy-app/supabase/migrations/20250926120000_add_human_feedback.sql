-- Add human feedback table for product deduplication
CREATE TABLE IF NOT EXISTS human_feedback (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    old_product TEXT NOT NULL,
    new_product TEXT NOT NULL,
    similarity_score FLOAT NOT NULL,
    human_decision TEXT NOT NULL CHECK (human_decision IN ('similar', 'different', 'duplicate', 'uncertain')),
    ml_prediction TEXT CHECK (ml_prediction IN ('similar', 'different')),
    reviewer_id UUID REFERENCES auth.users(id),
    comments TEXT,
    confidence_score FLOAT DEFAULT 0.0,
    processing_time INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add indexes for performance
CREATE INDEX idx_human_feedback_products ON human_feedback(old_product, new_product);
CREATE INDEX idx_human_feedback_reviewer ON human_feedback(reviewer_id) WHERE reviewer_id IS NOT NULL;
CREATE INDEX idx_human_feedback_decision ON human_feedback(human_decision);
CREATE INDEX idx_human_feedback_created_at ON human_feedback(created_at DESC);

-- Add RLS policies
ALTER TABLE human_feedback ENABLE ROW LEVEL SECURITY;

-- Allow authenticated users to read all feedback
CREATE POLICY "human_feedback_read" ON human_feedback FOR SELECT USING (auth.role() IS NOT NULL);

-- Allow authenticated users to insert their own feedback
CREATE POLICY "human_feedback_insert" ON human_feedback FOR INSERT WITH CHECK (auth.uid() = reviewer_id);

-- Allow users to update their own feedback
CREATE POLICY "human_feedback_update" ON human_feedback FOR UPDATE USING (auth.uid() = reviewer_id);

-- Add trigger for updated_at
CREATE TRIGGER update_human_feedback_updated_at 
    BEFORE UPDATE ON human_feedback 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Add audit trigger
CREATE TRIGGER audit_human_feedback 
    AFTER INSERT OR UPDATE OR DELETE ON human_feedback 
    FOR EACH ROW 
    EXECUTE FUNCTION audit_trigger_function();
