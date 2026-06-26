from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
import time
import logging
from datetime import datetime
import shutil
import pandas as pd
from pathlib import Path

from src.api.models import SystemHealth, APIConfig, CleanTextRequest, CleanTextResponse, CleanTextResult
from src.api.dependencies import app_state, create_job_id
from src.api.custom_websockets import notify_websockets, manager
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
            "pipeline": "operational" if app_state["pipeline"] else "initializing",
            "websockets": f"{len(manager.active_connections)} connected",
            "background_jobs": f"{len(app_state['jobs'])} jobs"
        }
    )

@router.get("/config", response_model=APIConfig)
async def get_config():
    """Get API configuration."""
    return app_state["config"]

@router.post("/config")
async def update_config(config: APIConfig):
    """Update API configuration."""
    app_state["config"] = config
    
    config_dict = config.model_dump() if hasattr(config, "model_dump") else config.dict()
    await notify_websockets({
        "type": "config_updated",
        "config": config_dict
    })
    return {"message": "Configuration updated successfully"}

@router.post("/upload/csv")
def upload_csv_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a CSV file for batch processing."""
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        # Save uploaded file
        upload_id = create_job_id()
        upload_path = str(BASE_DIR / "uploads" / f"{upload_id}_{file.filename}")
        
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
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

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
        logger.exception("Cleaning failed")
        raise HTTPException(status_code=500, detail=f"Cleaning failed: {str(e)}")
