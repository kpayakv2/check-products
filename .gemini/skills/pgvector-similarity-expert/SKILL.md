# pgvector Similarity Expert Skill

## Overview
Expert guidance for managing `pgvector` in PostgreSQL for high-accuracy product matching (384 dimensions).

## Key Procedures

### 1. Vector Indexing (CRITICAL)
- **HNSW Index:** Use HNSW (Hierarchical Navigable Small World) for fast, approximate nearest neighbor search on the `embedding` column.
- **IVFFlat:** Use IVFFlat for very large datasets where index build time is a concern.

### 2. Similarity Matching (HIGH)
- **Cosine Distance:** Use the `<=>` operator for cosine similarity (standard for sentence-transformers).
- **Thresholding:** Maintain a similarity threshold (e.g., 0.72) to balance between precision and recall.

### 3. Thai Language Preprocessing
- **Normalization:** Always use `ThaiTextProcessor` before generating embeddings to ensure consistency.

## How to Apply
Invoke this skill when:
- Tuning the similarity matching algorithm.
- Managing product embeddings and indexing performance.
- Investigating low-accuracy classification results.
