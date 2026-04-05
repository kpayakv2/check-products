-- Add embedding column to taxonomy_nodes
-- Required for hybrid classification functions

-- เพิ่ม embedding column ให้ taxonomy_nodes
ALTER TABLE taxonomy_nodes 
ADD COLUMN IF NOT EXISTS embedding vector(384);

-- สร้าง index สำหรับ vector search
CREATE INDEX IF NOT EXISTS idx_taxonomy_nodes_embedding 
ON taxonomy_nodes 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Comment
COMMENT ON COLUMN taxonomy_nodes.embedding IS 
'Vector embedding for category name (384-dim for local model paraphrase-multilingual-MiniLM-L12-v2)';

-- Log
DO $$
BEGIN
  RAISE NOTICE 'Added embedding column to taxonomy_nodes table';
  RAISE NOTICE 'Created vector index for fast similarity search';
END $$;
