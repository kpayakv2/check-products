from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import List
import os
from pathlib import Path
import logging

from src.api.models import JobStatus
from src.api.dependencies import app_state

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Jobs"])
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a background job."""
    if job_id not in app_state["jobs"]:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return app_state["jobs"][job_id]

@router.get("/jobs", response_model=List[JobStatus])
async def list_jobs():
    """List all background jobs."""
    return list(app_state["jobs"].values())

@router.get("/results/{job_id}")
async def get_job_results(job_id: str):
    """Get the results of a completed job."""
    if job_id not in app_state["jobs"]:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = app_state["jobs"][job_id]
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    results_file = str(BASE_DIR / "results" / f"batch_{job_id}.json")
    if not os.path.exists(results_file):
        raise HTTPException(status_code=404, detail="Results file not found")
    
    return FileResponse(results_file, media_type="application/json")
