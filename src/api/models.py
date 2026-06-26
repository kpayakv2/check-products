from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class ProductMatchRequest(BaseModel):
    """Single product matching request."""
    query_product: str = Field(..., description="Product name to match")
    reference_products: List[str] = Field(..., description="List of reference products")
    threshold: Optional[float] = Field(0.6, description="Similarity threshold")
    top_k: Optional[int] = Field(10, description="Maximum number of results")
    include_metadata: Optional[bool] = Field(True, description="Include processing metadata")
    include_confidence: Optional[bool] = Field(True, description="Include confidence scores")


class BatchMatchRequest(BaseModel):
    """Batch product matching request."""
    query_products: List[str] = Field(..., description="List of query products")
    reference_products: List[str] = Field(..., description="List of reference products")
    threshold: Optional[float] = Field(0.6, description="Similarity threshold")
    top_k: Optional[int] = Field(10, description="Maximum number of results per query")
    include_metadata: Optional[bool] = Field(True, description="Include processing metadata")
    include_confidence: Optional[bool] = Field(True, description="Include confidence scores")


class MatchResult(BaseModel):
    """Single match result."""
    query_product: str
    matched_product: str
    similarity_score: float
    rank: int
    confidence_score: Optional[float] = None
    confidence_level: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BatchMatchResponse(BaseModel):
    """Batch matching response."""
    job_id: str
    status: str
    total_queries: int
    processed: int
    matches_found: int
    processing_time: Optional[float] = None
    results: Optional[List[MatchResult]] = None
    performance_report: Optional[Dict[str, Any]] = None


class CleanTextRequest(BaseModel):
    """Batch text cleaning request."""
    texts: List[str] = Field(..., description="List of raw texts to clean")


class CleanTextResult(BaseModel):
    """Single cleaned text result."""
    original: str
    cleaned: str


class CleanTextResponse(BaseModel):
    """Batch cleaning response."""
    results: List[CleanTextResult]
    processing_time: float


class JobStatus(BaseModel):
    """Background job status."""
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: float = 0.0
    message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    results_url: Optional[str] = None


class SystemHealth(BaseModel):
    """System health check response."""
    status: str
    timestamp: datetime
    version: str
    uptime: float
    components: Dict[str, str]


class APIConfig(BaseModel):
    """API configuration."""
    default_threshold: float = 0.6
    default_top_k: int = 10
    max_batch_size: int = 1000
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_file_types: List[str] = ["csv", "xlsx", "json"]


class LearnVerifyRequest(BaseModel):
    """Request for auto-learning from verified product."""
    product_name: str = Field(..., description="The verified product name")
    category_id: str = Field(..., description="The confirmed category ID")


class EmbeddingRequest(BaseModel):
    """Single embedding request."""
    text: str = Field(..., description="Text to embed")


class BatchEmbeddingRequest(BaseModel):
    """Batch embedding request."""
    texts: List[str] = Field(..., description="Texts to embed")


class EmbeddingResponse(BaseModel):
    """Embedding response."""
    embedding: List[float]
    dimension: int
    model: str
    processing_time: float


class BatchEmbeddingResponse(BaseModel):
    """Batch embedding response."""
    embeddings: List[List[float]]
    count: int
    dimension: int
    model: str
    processing_time: float


class ScanInternalRequest(BaseModel):
    """Request to scan for internal duplicates."""
    threshold: Optional[float] = Field(0.95, description="Similarity threshold for matching")
    limit: Optional[int] = Field(100, description="Maximum number of duplicate pairs to return")


class InternalDuplicatePair(BaseModel):
    """A pair of internally duplicated products."""
    id_a: str
    name_a: str
    sku_a: str
    category_a: str
    category_id_a: str
    id_b: str
    name_b: str
    sku_b: str
    category_b: str
    category_id_b: str
    similarity: float
    ml_prediction: Optional[str] = None
    confidence: Optional[float] = None
    reason: Optional[str] = None


class ScanInternalResponse(BaseModel):
    """Response containing duplicate pairs found within the database."""
    total_scanned: int
    pairs_found: int
    results: List[InternalDuplicatePair]
    processing_time: float
    fallback_active: bool = False
    system_warnings: List[str] = Field(default_factory=list)


class FileDeduplicationRequest(BaseModel):
    """Request schema for uploading files deduplication."""
    old_products_path: str = Field(..., description="Path of existing products CSV in Supabase storage")
    new_products_path: str = Field(..., description="Path of new products CSV in Supabase storage")
    threshold: Optional[float] = Field(0.75, description="Similarity threshold for matching")


class ProductMatchResult(BaseModel):
    """Unified file-based product match result."""
    id: str
    newProduct: str
    oldProduct: str
    similarity: float
    confidence: float
    mlPrediction: str  # 'similar' or 'different'
    status: str = "pending"
    reason: str


class FileDeduplicationStats(BaseModel):
    """Statistics for file deduplication job."""
    totalNewProducts: int
    totalOldProducts: int
    needsReview: int
    autoApproved: int
    excludedDuplicates: int
    processingTime: float


class FileDeduplicationResponse(BaseModel):
    """Response containing matches, unique products, and job statistics."""
    success: bool
    results: List[ProductMatchResult]
    uniqueProducts: List[str]
    stats: FileDeduplicationStats
    summary: str


class ImportDeduplicationRequest(BaseModel):
    """Request schema for batch product deduplication against database."""
    products: List[str] = Field(..., description="List of product names to deduplicate")
    threshold: Optional[float] = Field(0.75, description="Similarity threshold")


