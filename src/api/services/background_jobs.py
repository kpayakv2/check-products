import time
import json
import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import logging

from src.api.models import BatchMatchRequest
from src.api.dependencies import app_state, initialize_pipeline
from src.api.custom_websockets import notify_websockets
from src.core.scoring_logic import enhance_matches
from src.main import generate_performance_report

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

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
            
            # Find matches for this batch (non-blocking)
            batch_matches = await asyncio.to_thread(
                pipeline.product_matcher.find_matches,
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
        
        # Enhance results (non-blocking)
        if request.include_metadata or request.include_confidence:
            enhanced_matches = await asyncio.to_thread(enhance_matches, all_matches)
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
        results_file = str(BASE_DIR / "results" / f"batch_{job_id}.json")
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
        logger.exception(f"Job {job_id} failed")
        job = app_state["jobs"][job_id]
        job.status = "failed"
        job.message = f"Job failed: {str(e)}"
        job.completed_at = datetime.now()
        
        await notify_websockets({
            "type": "batch_job_failed",
            "job_id": job_id,
            "error": str(e)
        })
