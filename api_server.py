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

# Import our Phase 4 pipeline
from fresh_architecture import ProductSimilarityPipeline
from fresh_implementations import ComponentFactory
from main_phase4 import Phase4Config, enhance_results, generate_performance_report


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
    "pipeline": None,
    "websocket_connections": set()
}


# =============================================================================
# WebSocket Connection Manager
# =============================================================================

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
    
    # Create necessary directories
    os.makedirs("temp", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("results", exist_ok=True)


@app.get("/", response_class=JSONResponse)
async def root():
    """API root endpoint."""
    return {
        "message": "Product Similarity API v5.0.0",
        "status": "operational",
        "docs": "/docs",
        "web_interface": "/web",
        "version": "5.0.0"
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
        performance_report = generate_performance_report(
            start_time, end_time, all_matches, 
            pipeline.product_matcher.config, request
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
