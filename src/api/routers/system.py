from fastapi import APIRouter, HTTPException
import time
import logging
from datetime import datetime
from pathlib import Path

from src.api.models import SystemHealth, CleanTextRequest, CleanTextResponse, CleanTextResult
from src.api.dependencies import app_state
from src.core.fresh_implementations import ComponentFactory

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["System"])
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

@router.get("/health", response_model=SystemHealth)
async def health_check():
    """System health check."""
    uptime = time.time() - app_state["start_time"]
    
    return SystemHealth(
        status="healthy",
        timestamp=datetime.now(),
        version="5.0.0",
        uptime=uptime,
        components={
            "pipeline": "operational" if app_state.get("pipeline") else "initializing",
            "embedding_model": "operational" if app_state.get("embedding_model") else "initializing",
            "taxonomy_service": "operational" if app_state.get("taxonomy_service") else "initializing",
        }
    )

@router.post("/clean", response_model=CleanTextResponse)
def clean_texts(request: CleanTextRequest):
    """
    Clean and normalize Thai product names.
    Used for Step 2 of the import wizard.
    """
    try:
        start_time = time.time()
        processor = ComponentFactory.create_text_processor("thai")
        
        results = []
        for text in request.texts:
            cleaned = processor.process(text)
            results.append(CleanTextResult(
                original=text,
                cleaned=cleaned
            ))
            
        processing_time = time.time() - start_time
        
        return CleanTextResponse(
            results=results,
            processing_time=round(processing_time, 3)
        )
    except Exception as e:
        logger.exception("Text cleaning failed")
        raise HTTPException(status_code=500, detail="Text cleaning failed")
