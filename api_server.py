#!/usr/bin/env python3
"""
Phase 5: FastAPI Backend for Product Similarity API
==================================================

Production-ready REST API with real-time capabilities.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import json
import time
import uuid
import os
from pathlib import Path
import pandas as pd
from datetime import datetime
import tempfile
import shutil
from types import SimpleNamespace

# Import our Phase 4 pipeline
from fresh_architecture import ProductSimilarityPipeline
from fresh_implementations import ComponentFactory
from main import Phase4Config, enhance_results, generate_performance_report

# Import embedding model
from advanced_models import SentenceTransformerModel

# Import Supabase for database access
try:
    from supabase import create_client, Client
except ImportError:
    print("WARNING: Supabase not installed. Category classification will be unavailable.")
    Client = None

import numpy as np
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path="taxonomy-app/.env.local")


# =============================================================================
# Pydantic Models for API
# =============================================================================

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


class JobStatus(BaseModel):
    """Background job status."""
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: float  # 0.0 to 1.0
    message: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    results_url: Optional[str] = None


class SystemHealth(BaseModel):
    """System health check response."""
    status: str
    timestamp: datetime
    version: str
    uptime: float
    components: Dict[str, str]


class CategorySuggestion(BaseModel):
    """Single category suggestion."""
    category_id: str
    category_name: str
    category_level: Optional[int] = 0
    confidence: float
    method: str
    matched_keyword: Optional[str] = None
    methods: Optional[List[str]] = None


class CategoryRequest(BaseModel):
    """Category classification request."""
    product_name: str = Field(..., description="Product name to classify")
    method: Optional[str] = Field("hybrid", description="Method: keyword, embedding, hybrid")
    top_k: Optional[int] = Field(3, description="Number of suggestions")


class BatchCategoryRequest(BaseModel):
    """Batch category classification request."""
    products: List[str] = Field(..., description="List of product names")
    method: Optional[str] = Field("hybrid", description="Method: keyword, embedding, hybrid")
    top_k: Optional[int] = Field(3, description="Number of suggestions per product")


class CategoryResponse(BaseModel):
    """Category classification response."""
    product_name: str
    suggestions: List[CategorySuggestion]
    top_suggestion: Optional[CategorySuggestion] = None
    processing_time: float


class BatchCategoryResponse(BaseModel):
    """Batch category classification response."""
    total_products: int
    results: List[CategoryResponse]
    processing_time: float


class APIConfig(BaseModel):
    """API configuration."""
    default_threshold: float = 0.6
    default_top_k: int = 10
    max_batch_size: int = 1000
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_file_types: List[str] = ["csv", "xlsx", "json"]


# =============================================================================
# FastAPI Application Setup
# =============================================================================

app = FastAPI(
    title="Product Similarity API",
    description="Advanced product matching API with AI-powered similarity detection",
    version="5.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for web interface
web_dir = Path(__file__).parent / "web"
if web_dir.exists():
    app.mount("/web", StaticFiles(directory=web_dir, html=True), name="web")
# Global state management
app_state = {
    "start_time": time.time(),
    "jobs": {},  # job_id -> JobStatus
    "config": APIConfig(),
    "pipeline": None,  # Lazy initialization
    "embedding_model": None,  # Lazy initialization for embeddings
    "category_classifier": None,  # Hybrid AI Classifier
    "supabase_client": None,  # Lazy initialization for Supabase
    "websocket_connections": set()
}

class ConnectionManager:
    """Manage WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove disconnected connections
                if connection in self.active_connections:
                    self.active_connections.remove(connection)


manager = ConnectionManager()


# =============================================================================
# Utility Functions
# =============================================================================

def initialize_pipeline():
    """Initialize the product similarity pipeline."""
    if app_state["pipeline"] is None:
        print("🔧 Initializing Phase 4 pipeline...")
        
        # Create configuration
        config = Phase4Config()
        config.model_name = "tfidf"
        config.similarity_method = "cosine"
        config.enable_performance_tracking = True
        config.include_metadata = True
        config.include_confidence_scores = True
        
        # Create components
        data_source = ComponentFactory.create_data_source("csv")
        data_sink = ComponentFactory.create_data_sink("csv")
        text_processor = ComponentFactory.create_text_processor("thai")
        embedding_model = ComponentFactory.create_embedding_model("tfidf")
        similarity_calculator = ComponentFactory.create_similarity_calculator("cosine")
        
        # Create matcher
        from fresh_architecture import ProductMatcher
        matcher = ProductMatcher(
            embedding_model=embedding_model,
            similarity_calculator=similarity_calculator,
            text_processor=text_processor,
            config=config
        )
        
        # Create pipeline
        pipeline = ProductSimilarityPipeline(
            data_source=data_source,
            data_sink=data_sink,
            product_matcher=matcher
        )
        
        app_state["pipeline"] = pipeline
        print("✅ Pipeline initialized successfully!")
    
    return app_state["pipeline"]


def create_job_id() -> str:
    """Create a unique job ID."""
    return str(uuid.uuid4())


async def notify_websockets(message: Dict[str, Any]):
    """Send message to all connected WebSocket clients."""
    if manager.active_connections:
        await manager.broadcast(json.dumps(message))


# =============================================================================
# API Endpoints
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    print("🚀 Starting Product Similarity API v5.0.0")
    initialize_pipeline()
    
    # Initialize Supabase connection
    initialize_supabase()

    # Initialize Category Classifier
    try:
        initialize_category_classifier()
    except Exception as e:
        print(f"⚠️ Failed to initialize Category Classifier: {e}")

    # Create necessary directories
    os.makedirs("output", exist_ok=True)


@app.post("/api/classify/category", response_model=CategoryResponse)
async def classify_category(request: CategoryRequest):
    """Classify a single product into a category."""
    start_time = time.time()
    
    classifier = initialize_category_classifier()
    if not classifier:
        raise HTTPException(status_code=503, detail="Category classifier not initialized")
    
    try:
        results = classifier.classify(request.product_name, method=request.method)
        suggestions = []
        
        for r in results[:request.top_k]:
            suggestions.append(CategorySuggestion(
                category_id=r['category_id'],
                category_name=r['category_name'],
                category_level=r.get('category_level', 0),
                confidence=r['confidence'],
                method=r['method'],
                matched_keyword=r.get('matched_keyword'),
                methods=r.get('methods')
            ))
        
        top_suggestion = suggestions[0] if suggestions else None
        
        return CategoryResponse(
            product_name=request.product_name,
            suggestions=suggestions,
            top_suggestion=top_suggestion,
            processing_time=time.time() - start_time
        )
    except Exception as e:
        print(f"❌ Error in classify_category: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/classify/batch", response_model=BatchCategoryResponse)
async def classify_category_batch(request: BatchCategoryRequest):
    """Classify multiple products into categories."""
    start_time = time.time()
    
    classifier = initialize_category_classifier()
    if not classifier:
        raise HTTPException(status_code=503, detail="Category classifier not initialized")
    
    results = []
    try:
        for product_name in request.products:
            p_start = time.time()
            res = classifier.classify(product_name, method=request.method)
            suggestions = []
            
            for r in res[:request.top_k]:
                suggestions.append(CategorySuggestion(
                    category_id=r['category_id'],
                    category_name=r['category_name'],
                    category_level=r.get('category_level', 0),
                    confidence=r['confidence'],
                    method=r['method'],
                    matched_keyword=r.get('matched_keyword'),
                    methods=r.get('methods')
                ))
            
            results.append(CategoryResponse(
                product_name=product_name,
                suggestions=suggestions,
                top_suggestion=suggestions[0] if suggestions else None,
                processing_time=time.time() - p_start
            ))
            
        return BatchCategoryResponse(
            total_products=len(request.products),
            results=results,
            processing_time=time.time() - start_time
        )
    except Exception as e:
        print(f"❌ Error in classify_category_batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "Product Similarity API (Legacy)",
        "version": "5.0.0",
        "status": "operational",
        "note": "This API is only for web_server.py (Flask). Use Supabase Edge Functions for new features.",
        "endpoints": {
            "health": "/api/v1/health",
            "embeddings": "/api/embed",
            "batch embeddings": "/api/embed/batch",
            "docs": "/docs",
            "web_interface": "/web"
        }
    }

@app.get("/api/v1/health", response_model=SystemHealth)
async def health_check():
    """System health check."""
    uptime = time.time() - app_state["start_time"]
    
    return SystemHealth(
        status="healthy",
        timestamp=datetime.now(),
        version="5.0.0",
        uptime=uptime,
        components={
            "pipeline": "operational" if app_state["pipeline"] else "initializing",
            "websockets": f"{len(manager.active_connections)} connected",
            "background_jobs": f"{len(app_state['jobs'])} jobs"
        }
    )


@app.get("/api/v1/config", response_model=APIConfig)
async def get_config():
    """Get API configuration."""
    return app_state["config"]


@app.post("/api/v1/config")
async def update_config(config: APIConfig):
    """Update API configuration."""
    app_state["config"] = config
    await notify_websockets({
        "type": "config_updated",
        "config": config.dict()
    })
    return {"message": "Configuration updated successfully"}


@app.post("/api/v1/match/single", response_model=List[MatchResult])
async def match_single_product(request: ProductMatchRequest):
    """Match a single product against reference products."""
    try:
        start_time = time.time()
        pipeline = initialize_pipeline()
        
        # Update pipeline configuration
        pipeline.product_matcher.config.similarity_threshold = request.threshold
        pipeline.product_matcher.config.top_k = request.top_k
        pipeline.product_matcher.config.include_metadata = request.include_metadata
        pipeline.product_matcher.config.include_confidence_scores = request.include_confidence
        
        # Find matches
        matches = pipeline.product_matcher.find_matches(
            query_products=[request.query_product],
            reference_products=request.reference_products
        )
        
        # Convert to API response format
        results = []
        for match in matches:
            result = MatchResult(
                query_product=match["query_product"],
                matched_product=match["matched_product"],
                similarity_score=match["similarity_score"],
                rank=match["rank"]
            )
            
            if request.include_confidence and "confidence_score" in match:
                result.confidence_score = match["confidence_score"]
                result.confidence_level = match.get("confidence_level", "unknown")
            
            if request.include_metadata:
                result.metadata = {
                    "processing_time": time.time() - start_time,
                    "processor_version": "phase5_api",
                    "timestamp": time.time()
                }
            
            results.append(result)
        
        # Notify WebSocket clients
        await notify_websockets({
            "type": "single_match_completed",
            "query_product": request.query_product,
            "matches_found": len(results),
            "processing_time": time.time() - start_time
        })
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Matching failed: {str(e)}")


@app.post("/api/v1/match/batch", response_model=BatchMatchResponse)
async def match_batch_products(request: BatchMatchRequest, background_tasks: BackgroundTasks):
    """Start a batch product matching job."""
    try:
        # Validate batch size
        if len(request.query_products) > app_state["config"].max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size exceeds maximum limit of {app_state['config'].max_batch_size}"
            )
        
        # Create job
        job_id = create_job_id()
        job_status = JobStatus(
            job_id=job_id,
            status="pending",
            progress=0.0,
            message="Job queued for processing",
            created_at=datetime.now()
        )
        
        app_state["jobs"][job_id] = job_status
        
        # Start background processing
        background_tasks.add_task(process_batch_job, job_id, request)
        
        # Notify WebSocket clients
        await notify_websockets({
            "type": "batch_job_started",
            "job_id": job_id,
            "total_queries": len(request.query_products)
        })
        
        return BatchMatchResponse(
            job_id=job_id,
            status="pending",
            total_queries=len(request.query_products),
            processed=0,
            matches_found=0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start batch job: {str(e)}")


async def process_batch_job(job_id: str, request: BatchMatchRequest):
    """Process a batch matching job in the background."""
    try:
        job = app_state["jobs"][job_id]
        job.status = "processing"
        job.message = "Processing batch matches..."
        
        start_time = time.time()
        pipeline = initialize_pipeline()
        
        # Update pipeline configuration
        pipeline.product_matcher.config.similarity_threshold = request.threshold
        pipeline.product_matcher.config.top_k = request.top_k
        pipeline.product_matcher.config.include_metadata = request.include_metadata
        pipeline.product_matcher.config.include_confidence_scores = request.include_confidence
        
        all_matches = []
        total_queries = len(request.query_products)
        
        # Process in batches for progress tracking
        batch_size = 50
        for i in range(0, total_queries, batch_size):
            batch_queries = request.query_products[i:i + batch_size]
            
            # Find matches for this batch
            batch_matches = pipeline.product_matcher.find_matches(
                query_products=batch_queries,
                reference_products=request.reference_products
            )
            
            all_matches.extend(batch_matches)
            
            # Update progress
            processed = min(i + batch_size, total_queries)
            job.progress = processed / total_queries
            job.message = f"Processed {processed}/{total_queries} queries"
            
            # Notify WebSocket clients
            await notify_websockets({
                "type": "batch_job_progress",
                "job_id": job_id,
                "progress": job.progress,
                "processed": processed,
                "total": total_queries
            })
        
        # Enhance results
        if request.include_metadata or request.include_confidence:
            enhanced_matches = enhance_results(all_matches, pipeline.product_matcher.config)
            all_matches = enhanced_matches
        
        # Generate performance report
        end_time = time.time()
        embedding_name = getattr(pipeline.product_matcher.config, 'model_name', None) or type(pipeline.product_matcher.embedding_model).__name__
        similarity_name = getattr(pipeline.product_matcher.config, 'similarity_method', None) or type(pipeline.product_matcher.similarity_calculator).__name__
        report_args = SimpleNamespace(model=embedding_name, similarity=similarity_name)
        performance_report = generate_performance_report(
            start_time, end_time, all_matches,
            pipeline.product_matcher.config, report_args
        )
        
        # Save results
        results_file = f"results/batch_{job_id}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "matches": all_matches,
                "performance_report": performance_report
            }, f, indent=2, ensure_ascii=False, default=str)
        
        # Complete job
        job.status = "completed"
        job.progress = 1.0
        job.message = f"Completed successfully. Found {len(all_matches)} matches."
        job.completed_at = datetime.now()
        job.results_url = f"/api/v1/results/{job_id}"
        
        # Notify WebSocket clients
        await notify_websockets({
            "type": "batch_job_completed",
            "job_id": job_id,
            "matches_found": len(all_matches),
            "processing_time": end_time - start_time
        })
        
    except Exception as e:
        job = app_state["jobs"][job_id]
        job.status = "failed"
        job.message = f"Job failed: {str(e)}"
        job.completed_at = datetime.now()
        
        await notify_websockets({
            "type": "batch_job_failed",
            "job_id": job_id,
            "error": str(e)
        })


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a background job."""
    if job_id not in app_state["jobs"]:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return app_state["jobs"][job_id]


@app.get("/api/v1/jobs", response_model=List[JobStatus])
async def list_jobs():
    """List all background jobs."""
    return list(app_state["jobs"].values())


@app.get("/api/v1/results/{job_id}")
async def get_job_results(job_id: str):
    """Get the results of a completed job."""
    if job_id not in app_state["jobs"]:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = app_state["jobs"][job_id]
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    results_file = f"results/batch_{job_id}.json"
    if not os.path.exists(results_file):
        raise HTTPException(status_code=404, detail="Results file not found")
    
    return FileResponse(results_file, media_type="application/json")


@app.post("/api/v1/upload/csv")
async def upload_csv_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a CSV file for batch processing."""
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        # Save uploaded file
        upload_id = create_job_id()
        upload_path = f"uploads/{upload_id}_{file.filename}"
        
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Parse CSV to extract products
        df = pd.read_csv(upload_path)
        
        # Try to detect product column
        product_columns = ["name", "product_name", "รายการ", "ชื่อสินค้า", "สินค้า"]
        product_column = None
        
        for col in product_columns:
            if col in df.columns:
                product_column = col
                break
        
        if product_column is None:
            # Use first column as default
            product_column = df.columns[0]
        
        products = df[product_column].dropna().tolist()
        
        return {
            "upload_id": upload_id,
            "filename": file.filename,
            "total_products": len(products),
            "product_column": product_column,
            "sample_products": products[:5],
            "upload_path": upload_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# =============================================================================
# Embedding API Endpoints (for Taxonomy Manager)
# =============================================================================

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


# =============================================================================
# Category Classification Models
# =============================================================================

class CategorySuggestion(BaseModel):
    """Single category suggestion."""
    category_id: str
    category_name: str
    category_level: int = 0
    confidence: float
    method: str
    matched_keyword: Optional[str] = None
    methods: Optional[List[str]] = None


class CategoryRequest(BaseModel):
    """Category classification request."""
    product_name: str = Field(..., description="Product name to classify")
    method: str = Field('hybrid', description="Classification method: keyword, embedding, or hybrid")
    top_k: int = Field(3, description="Number of suggestions to return")


class CategoryResponse(BaseModel):
    """Category classification response."""
    product_name: str
    suggestions: List[CategorySuggestion]
    top_suggestion: Optional[CategorySuggestion] = None
    processing_time: float


class BatchCategoryRequest(BaseModel):
    """Batch category classification request."""
    products: List[str] = Field(..., description="List of product names")
    method: str = Field('hybrid', description="Classification method")
    top_k: int = Field(3, description="Number of suggestions per product")


class BatchCategoryResponse(BaseModel):
    """Batch category classification response."""
    total_products: int
    results: List[CategoryResponse]
    processing_time: float


def initialize_embedding_model():
    """Initialize the embedding model (same as Product Similarity Checker)."""
    if app_state["embedding_model"] is None:
        print("🔧 Loading Sentence Transformer model...")
        try:
            model = SentenceTransformerModel(
                model_name="paraphrase-multilingual-MiniLM-L12-v2"
            )
            app_state["embedding_model"] = model
            print(f"✅ Model loaded! Dimension: {model.get_dimension()}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    return app_state["embedding_model"]


def initialize_supabase():
    """Initialize Supabase client."""
    if app_state["supabase_client"] is None and Client is not None:
        print("🔧 Connecting to Supabase...")
        try:
            supabase_url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

            if not supabase_url or not supabase_key:
                print("⚠️ Supabase credentials not found in environment variables.")
                return None

            client = create_client(supabase_url, supabase_key)
            app_state["supabase_client"] = client
            print(f"✅ Supabase connected: {supabase_url}")
        except Exception as e:
            print(f"⚠️ Failed to connect to Supabase: {e}")

    return app_state["supabase_client"]

class CategoryClassifier:
    """Category Classification Algorithm (from test_category_algorithm.py)."""
    
    def __init__(self, supabase_client: Client, embedding_model: SentenceTransformerModel):
        self.supabase = supabase_client
        self.embedding_model = embedding_model
        from fresh_implementations import ThaiTextProcessor
        self.processor = ThaiTextProcessor(normalize_numbers=True, normalize_thai_chars=True)
        self.taxonomy_flat = []
        self.keyword_rules = []
        self.synonyms = {}
        self.category_embeddings = {}
    
    def load_taxonomy(self):
        """โหลด Taxonomy จาก Supabase"""
        try:
            response = self.supabase.table("taxonomy_nodes").select("*").execute()
            nodes = response.data
            
            if not nodes:
                return False
            
            self.taxonomy_flat = nodes
            self._generate_category_embeddings()
            return True
        except Exception as e:
            print(f"❌ Failed to load taxonomy: {e}")
            return False
    
    def _generate_category_embeddings(self):
        """สร้าง Embeddings สำหรับหมวดหมู่"""
        category_texts = []
        category_ids = []
        
        for node in self.taxonomy_flat:
            # Clean category names for embedding
            text_th = self.processor.process(node['name_th'])
            text_parts = [text_th]
            if node.get('name_en'):
                text_parts.append(node['name_en'].lower())
            if node.get('keywords'):
                # Clean each keyword
                cleaned_kws = [self.processor.process(kw) for kw in node['keywords']]
                text_parts.extend(cleaned_kws)
            
            combined_text = " ".join(text_parts)
            category_texts.append(combined_text)
            category_ids.append(node['id'])
        
        embeddings = self.embedding_model.encode(category_texts)
        
        for cat_id, embedding in zip(category_ids, embeddings):
            self.category_embeddings[cat_id] = embedding
    
    def load_keyword_rules(self):
        """โหลด Keyword Rules จาก Supabase"""
        try:
            response = self.supabase.table("keyword_rules").select("*").eq("is_active", True).execute()
            self.keyword_rules = response.data
            return True
        except Exception as e:
            print(f"⚠️ Failed to load keyword rules: {e}")
            return False
    
    def load_synonyms(self):
        """โหลด Synonyms จาก Supabase"""
        try:
            response = self.supabase.table("synonym_lemmas")\
                .select("*, synonym_terms(*)")\
                .eq("is_verified", True)\
                .execute()
            
            for lemma in response.data:
                if lemma.get('synonym_terms'):
                    for term in lemma['synonym_terms']:
                        self.synonyms[term['term']] = lemma['lemma_id'] # Fix to lemma_id
            return True
        except Exception as e:
            print(f"⚠️ Failed to load synonyms: {e}")
            return False

    def classify_keyword_based(self, product_name: str, top_k: int = 5) -> List[Dict]:
        """จัดหมวดหมู่ด้วย Keyword Matching"""
        # Clean product name before matching
        clean_name = self.processor.process(product_name)
        matches = {}
        
        for rule in self.keyword_rules:
            keywords = rule.get('keywords', [])
            for keyword in keywords:
                # Clean keyword for match consistency
                clean_kw = self.processor.process(keyword)
                if clean_kw in clean_name:
                    cat_info = next((n for n in self.taxonomy_flat if n['id'] == rule['category_id']), None)
                    cat_id = rule['category_id']
                    confidence = rule.get('priority', 1) * 0.1
                    
                    if cat_id not in matches or matches[cat_id]['confidence'] < confidence:
                        match_data = {
                            'category_id': cat_id,
                            'method': 'keyword_rule',
                            'matched_keyword': keyword,
                            'confidence': confidence
                        }
                        if cat_info:
                            match_data['category_name'] = cat_info['name_th']
                            match_data['category_level'] = cat_info.get('level', 0)
                        matches[cat_id] = match_data
        
        for node in self.taxonomy_flat:
            if node.get('keywords'):
                for keyword in node['keywords']:
                    clean_kw = self.processor.process(keyword)
                    if clean_kw in clean_name:
                        cat_id = node['id']
                        if cat_id not in matches or matches[cat_id]['confidence'] < 0.7:
                            matches[cat_id] = {
                                'category_id': cat_id,
                                'category_name': node['name_th'],
                                'category_level': node.get('level', 0),
                                'method': 'taxonomy_keyword',
                                'matched_keyword': keyword,
                                'confidence': 0.7
                            }
        
        results = list(matches.values())
        results.sort(key=lambda x: (-x['confidence'], x.get('category_level', 0)))
        return results[:top_k]

    def classify_embedding_based(self, product_name: str, top_k: int = 5) -> List[Dict]:
        """จัดหมวดหมู่ด้วย Embedding Similarity"""
        if not self.category_embeddings:
            return []
        
        # Clean product name before embedding
        clean_name = self.processor.process(product_name)
        product_embedding = self.embedding_model.encode([clean_name])[0]
        similarities = []
        
        for cat_id, cat_embedding in self.category_embeddings.items():
            similarity = np.dot(product_embedding, cat_embedding) / (
                np.linalg.norm(product_embedding) * np.linalg.norm(cat_embedding)
            )
            cat_info = next((n for n in self.taxonomy_flat if n['id'] == cat_id), None)
            if cat_info:
                similarities.append({
                    'category_id': cat_id,
                    'category_name': cat_info['name_th'],
                    'category_level': cat_info.get('level', 0),
                    'method': 'embedding',
                    'confidence': float(similarity)
                })
        
        similarities.sort(key=lambda x: x['confidence'], reverse=True)
        return similarities[:top_k]

    def classify_hybrid(self, product_name: str, top_k: int = 3) -> List[Dict]:
        """จัดหมวดหมู่ด้วย Hybrid Approach (Keyword + Embedding)"""
        # Clean product name once at the start
        clean_name = self.processor.process(product_name)
        
        keyword_matches = self.classify_keyword_based(clean_name)
        embedding_matches = self.classify_embedding_based(clean_name, top_k=10)
        
        combined = {}
        for match in keyword_matches:
            cat_id = match['category_id']
            combined[cat_id] = match.copy()
            combined[cat_id]['methods'] = [match['method']]
        
        for match in embedding_matches:
            cat_id = match['category_id']
            if cat_id not in combined:
                combined[cat_id] = match.copy()
                combined[cat_id]['methods'] = [match['method']]
            else:
                combined[cat_id]['confidence'] = (combined[cat_id]['confidence'] * 0.6 + match['confidence'] * 0.4)
                if match['method'] not in combined[cat_id]['methods']:
                    combined[cat_id]['methods'].append(match['method'])
        
        results = list(combined.values())
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results[:top_k]

    def classify(self, product_name: str, method: str = 'hybrid') -> List[Dict]:
        """Main classification method"""
        # Clean name is handled inside each sub-method
        if method == 'keyword':
            return self.classify_keyword_based(product_name)
        elif method == 'embedding':
            return self.classify_embedding_based(product_name)
        else:
            return self.classify_hybrid(product_name)


def initialize_category_classifier():
    """Initialize the category classifier."""
    if app_state["category_classifier"] is None:
        supabase = initialize_supabase()
        model = initialize_embedding_model()
        
        if supabase and model:
            print("🔧 Initializing Category Classifier...")
            classifier = CategoryClassifier(supabase, model)
            if classifier.load_taxonomy():
                classifier.load_keyword_rules()
                classifier.load_synonyms()
                app_state["category_classifier"] = classifier
                print("✅ Category Classifier initialized!")
            else:
                print("⚠️ Failed to initialize Category Classifier (taxonomy empty)")
    
    return app_state["category_classifier"]


# Category Classifier removed - use Supabase Edge Functions instead
# See: supabase/functions/hybrid-classification-local/


# CategoryClassifier class removed - use Supabase Edge Functions instead
# See: supabase/functions/hybrid-classification-local/


@app.post("/api/embed", response_model=EmbeddingResponse)
async def embed_single(request: EmbeddingRequest):
    """
    Generate embedding for a single text.
    Uses the same model as Product Similarity Checker for consistency.
    """
    try:
        start_time = time.time()
        model = initialize_embedding_model()
        
        # Generate embedding (SentenceTransformerModel uses 'encode' method)
        embeddings = model.encode([request.text])
        embedding = embeddings[0].tolist()
        
        processing_time = time.time() - start_time
        
        return EmbeddingResponse(
            embedding=embedding,
            dimension=len(embedding),
            model=model.model_name,
            processing_time=round(processing_time, 3)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")


@app.post("/api/embed/batch", response_model=BatchEmbeddingResponse)
async def embed_batch(request: BatchEmbeddingRequest):
    """Generate embeddings for multiple texts (batch processing)."""
    try:
        start_time = time.time()
        model = initialize_embedding_model()
        
        # Generate embeddings (SentenceTransformerModel uses 'encode' method)
        embeddings = model.encode(request.texts)
        embeddings_list = [emb.tolist() for emb in embeddings]
        
        processing_time = time.time() - start_time
        
        return BatchEmbeddingResponse(
            embeddings=embeddings_list,
            count=len(embeddings_list),
            dimension=len(embeddings_list[0]) if embeddings_list else 0,
            model=model.model_name,
            processing_time=round(processing_time, 3)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch embedding failed: {str(e)}")


# Classification endpoints removed - use Supabase Edge Functions instead
# See: supabase/functions/hybrid-classification-local/


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# =============================================================================
# Main Application Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Product Similarity API Server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🌐 Web Interface: http://localhost:8000/web")
    print("📊 Health Check: http://localhost:8000/api/v1/health")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
