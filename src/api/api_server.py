#!/usr/bin/env python3
"""
Phase 5: FastAPI Backend for Product Similarity API
==================================================

Production-ready REST API with real-time capabilities.
"""

import os
import sys

# Reconfigure stdout/stderr to replace characters that cannot be encoded in the local terminal's codepage (e.g. cp874/cp1252)
# This prevents UnicodeEncodeError crashes when printing emojis or special characters on Windows command prompts.
try:
    sys.stdout.reconfigure(errors='replace')
    sys.stderr.reconfigure(errors='replace')
except AttributeError:
    pass

from pathlib import Path
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load environment variables robustly
env_path = BASE_DIR / "taxonomy-app" / ".env.local"
load_dotenv(dotenv_path=str(env_path))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.dependencies import initialize_pipeline, initialize_embedding_model
from src.api.custom_websockets import manager
from src.api.routers import match, embed, system, jobs, learn, internal_match, classify


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


# =============================================================================
# API Endpoints & Routers
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    print("[PHAYAK] Starting Product Similarity API v5.0.0")
    initialize_pipeline()
    initialize_embedding_model()
    
    # Create necessary directories
    os.makedirs(BASE_DIR / "results", exist_ok=True)
    os.makedirs(BASE_DIR / "uploads", exist_ok=True)


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "Product Similarity API",
        "version": "5.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/api/v1/health",
            "embeddings": "/api/embed",
            "batch embeddings": "/api/embed/batch",
            "docs": "/docs",
            "web_interface": "/web"
        }
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Include Routers
app.include_router(system.router)
app.include_router(match.router)
app.include_router(embed.router)
app.include_router(jobs.router)
app.include_router(learn.router)
app.include_router(internal_match.router)
app.include_router(classify.router)


if __name__ == "__main__":
    import uvicorn
    
    print("[PHAYAK] Starting Product Similarity API Server (Local Provider Mode)...")
    print("[DOCS] API Documentation: http://127.0.0.1:8000/docs")
    print("[HEALTH] Health Check: http://127.0.0.1:8000/api/v1/health")
    
    uvicorn.run(
        "src.api.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
