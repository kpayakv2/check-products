from fastapi import APIRouter, HTTPException, BackgroundTasks
import time
import asyncio
from typing import List
import logging
from datetime import datetime

from src.api.models import ProductMatchRequest, MatchResult, BatchMatchRequest, BatchMatchResponse, JobStatus
from src.api.dependencies import initialize_pipeline, create_job_id, app_state
from src.api.custom_websockets import notify_websockets
from src.api.services.background_jobs import process_batch_job

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/match", tags=["Matching"])

@router.post("/single", response_model=List[MatchResult])
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
        
        # Find matches using thread to avoid blocking event loop
        matches = await asyncio.to_thread(
            pipeline.product_matcher.find_matches,
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
        logger.exception("Matching failed")
        raise HTTPException(status_code=500, detail=f"Matching failed: {str(e)}")

@router.post("/batch", response_model=BatchMatchResponse)
async def match_batch_products(request: BatchMatchRequest, background_tasks: BackgroundTasks):
    """Start a batch product matching job."""
    try:
        # Validate batch size
        if len(request.query_products) > app_state["config"].max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size exceeds maximum limit of {app_state['config'].max_batch_size}"
            )
        
        # Cleanup old jobs to prevent memory leak (older than 1 hour)
        current_time = datetime.now()
        expired_jobs = [jid for jid, j in app_state["jobs"].items() 
                        if j.completed_at and (current_time - j.completed_at).total_seconds() > 3600]
        for jid in expired_jobs:
            del app_state["jobs"][jid]

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
        logger.exception("Failed to start batch job")
        raise HTTPException(status_code=500, detail=f"Failed to start batch job: {str(e)}")
