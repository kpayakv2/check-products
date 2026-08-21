from fastapi import APIRouter, HTTPException
import time
import logging
import numpy as np
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from src.api.dependencies import get_taxonomy_service, initialize_embedding_model
from src.core.scoring_logic import calculate_hybrid_score

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/classify", tags=["Classification"])


# =============================================================================
# Pydantic Models
# =============================================================================

class CategoryRequest(BaseModel):
    """Category classification request."""
    product_name: str = Field(..., description="Product name to classify")
    method: str = Field("hybrid", description="Classification method: keyword, embedding, or hybrid")
    top_k: int = Field(3, description="Number of suggestions to return")


class BatchCategoryRequest(BaseModel):
    """Batch category classification request."""
    products: List[str] = Field(..., description="List of product names")
    method: str = Field("hybrid", description="Classification method")
    top_k: int = Field(3, description="Number of suggestions per product")


class CategorySuggestion(BaseModel):
    """Single category suggestion."""
    category_id: str
    category_name: str
    confidence: float
    method: str
    matched_keyword: Optional[str] = None


class CategoryResponse(BaseModel):
    """Category classification response."""
    product_name: str
    category_id: Optional[str] = None
    category_name: Optional[str] = None
    confidence: float = 0.0
    suggestions: List[CategorySuggestion] = []
    processing_time: float


class BatchCategoryResponse(BaseModel):
    """Batch category classification response."""
    total_products: int
    results: List[CategoryResponse]
    processing_time: float


# =============================================================================
# Classification Logic
# =============================================================================

def _classify_keyword_based(product_name: str, keyword_rules: list, category_names: dict, top_k: int = 5) -> List[Dict]:
    """
    Keyword-based classification.
    วนหา keyword_rules ที่ match ชื่อสินค้า แล้ว score ตาม priority + จำนวน match
    """
    product_lower = product_name.lower()
    scores: Dict[str, float] = {}
    matched_kw: Dict[str, str] = {}

    for rule in keyword_rules:
        if not rule.get("is_active", True):
            continue

        cat_id = rule.get("category_id")
        if not cat_id:
            continue

        keywords = rule.get("keywords") or []
        priority = rule.get("priority", 5)  # 1=highest, 10=lowest

        # score per rule = จำนวน keyword ที่ match / priority
        match_count = 0
        first_match = None
        for kw in keywords:
            if kw and kw.lower() in product_lower:
                match_count += 1
                if first_match is None:
                    first_match = kw

        if match_count > 0:
            rule_score = match_count / max(priority, 1)
            current = scores.get(cat_id, 0.0)
            if rule_score > current:
                scores[cat_id] = rule_score
                matched_kw[cat_id] = first_match or ""

    if not scores:
        return []

    # Normalize scores to [0, 1]
    max_score = max(scores.values())
    results = []
    for cat_id, raw_score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
        confidence = min(raw_score / max(max_score, 1e-9), 1.0)
        results.append({
            "category_id": cat_id,
            "category_name": category_names.get(cat_id, "Unknown"),
            "confidence": round(confidence, 4),
            "method": "keyword",
            "matched_keyword": matched_kw.get(cat_id),
        })

    return results


def _classify_embedding_based(product_name: str, category_embeddings: dict, category_names: dict,
                               embedding_model, top_k: int = 5) -> List[Dict]:
    """
    Embedding-based classification.
    Encode ชื่อสินค้า → 384-dim → cosine similarity กับ category_embeddings
    """
    if not category_embeddings:
        return []

    # Encode product
    try:
        product_emb = embedding_model.encode([product_name])[0]
    except Exception as e:
        logger.error(f"Embedding encode failed: {e}")
        return []

    # Compute cosine similarity vs all categories
    product_norm = np.linalg.norm(product_emb)
    if product_norm == 0:
        return []
    product_emb_normalized = product_emb / product_norm

    scores = []
    for cat_id, cat_emb in category_embeddings.items():
        cat_norm = np.linalg.norm(cat_emb)
        if cat_norm == 0:
            continue
        cat_emb_normalized = cat_emb / cat_norm
        sim = float(np.dot(product_emb_normalized, cat_emb_normalized))
        scores.append((cat_id, sim))

    # Sort and return top_k
    scores.sort(key=lambda x: x[1], reverse=True)
    results = []
    for cat_id, sim in scores[:top_k]:
        results.append({
            "category_id": cat_id,
            "category_name": category_names.get(cat_id, "Unknown"),
            "confidence": round(max(sim, 0.0), 4),
            "method": "embedding",
            "matched_keyword": None,
        })

    return results


def _classify_hybrid(product_name: str, keyword_rules: list, category_embeddings: dict,
                     category_names: dict, embedding_model, top_k: int = 3) -> List[Dict]:
    """
    Hybrid classification: Keyword 60% + Embedding 40% (see src/core/scoring_logic.py)
    """
    kw_results = _classify_keyword_based(product_name, keyword_rules, category_names, top_k=top_k * 2)
    emb_results = _classify_embedding_based(product_name, category_embeddings, category_names,
                                            embedding_model, top_k=top_k * 2)

    # Merge scores by category_id
    combined: Dict[str, Dict] = {}

    for r in kw_results:
        cid = r["category_id"]
        combined[cid] = {
            "category_id": cid,
            "category_name": r["category_name"],
            "kw_score": r["confidence"],
            "emb_score": 0.0,
            "matched_keyword": r.get("matched_keyword"),
        }

    for r in emb_results:
        cid = r["category_id"]
        if cid in combined:
            combined[cid]["emb_score"] = r["confidence"]
        else:
            combined[cid] = {
                "category_id": cid,
                "category_name": r["category_name"],
                "kw_score": 0.0,
                "emb_score": r["confidence"],
                "matched_keyword": None,
            }

    # Compute hybrid score
    results = []
    for cid, data in combined.items():
        hybrid_score = calculate_hybrid_score(data["kw_score"], data["emb_score"])
        results.append({
            "category_id": cid,
            "category_name": data["category_name"],
            "confidence": round(hybrid_score, 4),
            "method": "hybrid",
            "matched_keyword": data.get("matched_keyword"),
        })

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results[:top_k]


def _run_classification(product_name: str, method: str, top_k: int,
                        taxonomy_service, embedding_model) -> List[Dict]:
    """Entry point for classification. Loads data lazily if needed."""
    # Ensure taxonomy data is loaded
    if not taxonomy_service.category_names:
        taxonomy_service.load_taxonomy_nodes()
    if not taxonomy_service.keyword_rules:
        taxonomy_service.load_keyword_rules()

    kw_rules = taxonomy_service.keyword_rules
    cat_embs = taxonomy_service.category_embeddings
    cat_names = taxonomy_service.category_names

    if method == "keyword":
        return _classify_keyword_based(product_name, kw_rules, cat_names, top_k)
    elif method == "embedding":
        return _classify_embedding_based(product_name, cat_embs, cat_names, embedding_model, top_k)
    else:  # hybrid (default)
        return _classify_hybrid(product_name, kw_rules, cat_embs, cat_names, embedding_model, top_k)


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/category", response_model=CategoryResponse)
async def classify_category(request: CategoryRequest):
    """
    Classify a product into a taxonomy category.

    Methods:
    - keyword : Keyword matching only (fast, rule-based)
    - embedding: Vector similarity only (semantic)
    - hybrid  : Keyword 60% + Embedding 40% (default, most accurate)
    """
    start_time = time.time()
    try:
        taxonomy_service = get_taxonomy_service()
        if not taxonomy_service:
            raise HTTPException(status_code=503, detail="TaxonomyService unavailable — Supabase not connected")

        embedding_model = initialize_embedding_model()

        suggestions_raw = _run_classification(
            product_name=request.product_name,
            method=request.method,
            top_k=request.top_k,
            taxonomy_service=taxonomy_service,
            embedding_model=embedding_model,
        )

        suggestions = [CategorySuggestion(**s) for s in suggestions_raw]

        top = suggestions[0] if suggestions else None

        return CategoryResponse(
            product_name=request.product_name,
            category_id=top.category_id if top else None,
            category_name=top.category_name if top else None,
            confidence=top.confidence if top else 0.0,
            suggestions=suggestions,
            processing_time=round(time.time() - start_time, 3),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Category classification failed")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@router.post("/batch", response_model=BatchCategoryResponse)
async def classify_batch(request: BatchCategoryRequest):
    """Classify multiple products in batch."""
    start_time = time.time()
    try:
        taxonomy_service = get_taxonomy_service()
        if not taxonomy_service:
            raise HTTPException(status_code=503, detail="TaxonomyService unavailable")

        embedding_model = initialize_embedding_model()

        # Pre-load taxonomy once for the whole batch
        if not taxonomy_service.category_names:
            taxonomy_service.load_taxonomy_nodes()
        if not taxonomy_service.keyword_rules:
            taxonomy_service.load_keyword_rules()

        results = []
        for product_name in request.products:
            t0 = time.time()
            suggestions_raw = _run_classification(
                product_name=product_name,
                method=request.method,
                top_k=request.top_k,
                taxonomy_service=taxonomy_service,
                embedding_model=embedding_model,
            )
            suggestions = [CategorySuggestion(**s) for s in suggestions_raw]
            top = suggestions[0] if suggestions else None

            results.append(CategoryResponse(
                product_name=product_name,
                category_id=top.category_id if top else None,
                category_name=top.category_name if top else None,
                confidence=top.confidence if top else 0.0,
                suggestions=suggestions,
                processing_time=round(time.time() - t0, 3),
            ))

        return BatchCategoryResponse(
            total_products=len(request.products),
            results=results,
            processing_time=round(time.time() - start_time, 3),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Batch classification failed")
        raise HTTPException(status_code=500, detail=f"Batch classification failed: {str(e)}")
