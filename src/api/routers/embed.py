from fastapi import APIRouter, HTTPException
import time
import logging

from src.api.models import EmbeddingRequest, BatchEmbeddingRequest, EmbeddingResponse, BatchEmbeddingResponse
from src.api.dependencies import initialize_embedding_model

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/embed", tags=["Embeddings"])

@router.post("", response_model=EmbeddingResponse)
def embed_single(request: EmbeddingRequest):
    """
    Generate embedding for a single text.
    Primary use: Called by Supabase Edge Functions.
    """
    try:
        start_time = time.time()
        model = initialize_embedding_model()
        
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
        logger.exception("Embedding failed")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

@router.post("/batch", response_model=BatchEmbeddingResponse)
def embed_batch(request: BatchEmbeddingRequest):
    """Generate embeddings for multiple texts."""
    try:
        start_time = time.time()
        model = initialize_embedding_model()
        
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
        logger.exception("Batch embedding failed")
        raise HTTPException(status_code=500, detail=f"Batch embedding failed: {str(e)}")
