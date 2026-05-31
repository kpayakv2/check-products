from fastapi import APIRouter, HTTPException
import time
import os
import json
import logging
import numpy as np
from typing import List

from src.api.models import ScanInternalRequest, ScanInternalResponse, InternalDuplicatePair
from src.api.dependencies import get_taxonomy_service, get_ml_learning_system
from src.services.human_feedback_system import ProductComparison, FeedbackType

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/match", tags=["Matching"])

@router.post("/scan-internal", response_model=ScanInternalResponse)
async def scan_internal_duplicates(request: ScanInternalRequest):
    """
    Scan approved products in the database for duplicates using pairwise cosine similarity.
    Threshold default is 0.95.
    """
    start_time = time.time()
    try:
        # Get Taxonomy Service to obtain Supabase Client
        service = get_taxonomy_service()
        if not service or not service.supabase:
            raise HTTPException(status_code=503, detail="Supabase connection unavailable")
            
        supabase = service.supabase
        ml_system = get_ml_learning_system()
        
        # 1. Fetch Taxonomy Category Names Mapping
        logger.info("Fetching taxonomy category mapping...")
        cat_res = supabase.table("taxonomy_nodes").select("id, name_th").execute()
        category_mapping = {cat["id"]: cat["name_th"] for cat in cat_res.data}
        
        # 1.5 Fetch reviewed matches to prevent showing them again
        logger.info("Fetching reviewed similarity matches...")
        reviewed_res = supabase.table("similarity_matches")\
                               .select("product_a_id, product_b_id")\
                               .eq("reviewed", True)\
                               .execute()
        
        reviewed_pairs = set()
        if reviewed_res.data:
            for m in reviewed_res.data:
                # Sort IDs to guarantee consistent lookup keys (a_id < b_id)
                pair = tuple(sorted([m["product_a_id"], m["product_b_id"]]))
                reviewed_pairs.add(pair)
        logger.info(f"Loaded {len(reviewed_pairs)} reviewed pairs from database.")
        
        # 2. Fetch all Approved Products in pages
        logger.info("Fetching approved products...")
        all_products = []
        page_size = 1000
        offset = 0
        
        while True:
            res = supabase.table("products")\
                          .select("id, name_th, sku, embedding, category_id")\
                          .eq("status", "approved")\
                          .range(offset, offset + page_size - 1)\
                          .execute()
            
            data = res.data
            if not data:
                break
                
            all_products.extend(data)
            offset += len(data)
            
            if len(data) < page_size:
                break
                
        logger.info(f"Loaded {len(all_products)} approved products for scanning.")
        
        # 3. Filter products with valid embeddings
        products_with_emb = [p for p in all_products if p.get('embedding') is not None]
        logger.info(f"Products with valid embeddings: {len(products_with_emb)}")
        
        if len(products_with_emb) < 2:
            return ScanInternalResponse(
                total_scanned=len(all_products),
                pairs_found=0,
                results=[],
                processing_time=time.time() - start_time
            )
            
        # 4. Extract data and build embeddings matrix
        ids = [p['id'] for p in products_with_emb]
        names = [p['name_th'] for p in products_with_emb]
        skus = [p['sku'] or "N/A" for p in products_with_emb]
        cat_ids = [p['category_id'] or "" for p in products_with_emb]
        
        emb_matrix = []
        for p in products_with_emb:
            emb = p['embedding']
            if isinstance(emb, str):
                emb = json.loads(emb)
            emb_matrix.append(emb)
            
        emb_matrix = np.array(emb_matrix, dtype=np.float32)
        
        # 5. Normalize embeddings for fast Cosine Similarity via dot product
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_emb = emb_matrix / norms
        
        # Compute similarity matrix
        similarity_matrix = np.dot(normalized_emb, normalized_emb.T)
        
        # VECTORIZED FILTERING:
        # Get upper triangle mask to avoid self-matching and duplicate pairs (a, b) and (b, a)
        triu_mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
        
        # Find elements exceeding the threshold in the upper triangle
        match_mask = (similarity_matrix >= request.threshold) & triu_mask
        i_indices, j_indices = np.where(match_mask)
        
        logger.info(f"Vectorized filtering found {len(i_indices)} candidate duplicate pairs above threshold {request.threshold}.")
        
        # 6. Process candidate pairs and apply batch ML inference
        duplicate_pairs = []
        fallback_active = False
        system_warnings = []
        
        # --- STAGE 2: Batch ML Inference ---
        if ml_system and ml_system.model.is_trained and len(i_indices) > 0:
            try:
                logger.info("Pre-cleaning all product names for batch ML inference...")
                cleaned_names = [ml_system.text_processor.process(name) for name in names]
                
                comparisons = []
                candidate_indices = []
                
                for idx in range(len(i_indices)):
                    i, j = int(i_indices[idx]), int(j_indices[idx])
                    sim = float(similarity_matrix[i, j])
                    
                    # Check if this pair has already been reviewed
                    pair_key = tuple(sorted([ids[i], ids[j]]))
                    if pair_key in reviewed_pairs:
                        continue # Skip already resolved pairs
                        
                    comp = ProductComparison(
                        id="inference",
                        product1=names[i],
                        product2=names[j],
                        product1_cleaned=cleaned_names[i],
                        product2_cleaned=cleaned_names[j],
                        similarity_score=sim,
                        confidence_score=0.8,
                        ml_prediction=FeedbackType.SIMILAR
                    )
                    comparisons.append(comp)
                    candidate_indices.append((i, j, sim))
                    
                if comparisons:
                    logger.info(f"Running batch ML inference on {len(comparisons)} candidate pairs...")
                    # Extract features for all comparisons in one vectorized call
                    X = ml_system.model.feature_extractor.extract_features(comparisons)
                    
                    # Run predictions in batch
                    predictions = ml_system.model.model.predict(X)
                    probabilities = ml_system.model.model.predict_proba(X)
                    predicted_labels = ml_system.model.label_encoder.inverse_transform(predictions)
                    
                    for idx, (i, j, sim) in enumerate(candidate_indices):
                        pred_label = predicted_labels[idx]
                        conf = float(max(probabilities[idx]))
                        
                        if pred_label == FeedbackType.DIFFERENT.value and conf > 0.6:
                            # ML model highly confident they are different. Filter it out.
                            continue
                            
                        cat_a = category_mapping.get(cat_ids[i], "Unmapped")
                        cat_b = category_mapping.get(cat_ids[j], "Unmapped")
                        
                        duplicate_pairs.append(InternalDuplicatePair(
                            id_a=ids[i],
                            name_a=names[i],
                            sku_a=skus[i],
                            category_a=cat_a,
                            category_id_a=cat_ids[i],
                            id_b=ids[j],
                            name_b=names[j],
                            sku_b=skus[j],
                            category_b=cat_b,
                            category_id_b=cat_ids[j],
                            similarity=round(sim, 4)
                        ))
            except Exception as e:
                logger.exception("Batch ML Inference failed. Falling back to non-ML scan.")
                fallback_active = True
                system_warnings.append("ระบบ AI ขัดข้อง กำลังใช้ระบบสำรอง (Cosine Similarity)")
                # Fallback in case batch ML prediction fails: process without ML filtering
                for idx in range(len(i_indices)):
                    i, j = int(i_indices[idx]), int(j_indices[idx])
                    sim = float(similarity_matrix[i, j])
                    
                    pair_key = tuple(sorted([ids[i], ids[j]]))
                    if pair_key in reviewed_pairs:
                        continue
                        
                    cat_a = category_mapping.get(cat_ids[i], "Unmapped")
                    cat_b = category_mapping.get(cat_ids[j], "Unmapped")
                    
                    duplicate_pairs.append(InternalDuplicatePair(
                        id_a=ids[i],
                        name_a=names[i],
                        sku_a=skus[i],
                        category_a=cat_a,
                        category_id_a=cat_ids[i],
                        id_b=ids[j],
                        name_b=names[j],
                        sku_b=skus[j],
                        category_b=cat_b,
                        category_id_b=cat_ids[j],
                        similarity=round(sim, 4)
                    ))
        else:
            # Non-ML fallback / model not trained
            fallback_active = True
            system_warnings.append("AI Model ยังไม่พร้อมใช้งาน กำลังใช้ระบบสำรอง (Cosine Similarity)")
            for idx in range(len(i_indices)):
                i, j = int(i_indices[idx]), int(j_indices[idx])
                sim = float(similarity_matrix[i, j])
                
                pair_key = tuple(sorted([ids[i], ids[j]]))
                if pair_key in reviewed_pairs:
                    continue
                    
                cat_a = category_mapping.get(cat_ids[i], "Unmapped")
                cat_b = category_mapping.get(cat_ids[j], "Unmapped")
                
                duplicate_pairs.append(InternalDuplicatePair(
                    id_a=ids[i],
                    name_a=names[i],
                    sku_a=skus[i],
                    category_a=cat_a,
                    category_id_a=cat_ids[i],
                    id_b=ids[j],
                    name_b=names[j],
                    sku_b=skus[j],
                    category_b=cat_b,
                    category_id_b=cat_ids[j],
                    similarity=round(sim, 4)
                ))
            
        # Sort pairs by similarity descending
        duplicate_pairs.sort(key=lambda x: x.similarity, reverse=True)
        
        # Apply limit
        results = duplicate_pairs[:request.limit]
        
        processing_time = time.time() - start_time
        logger.info(f"Scan complete. Found {len(duplicate_pairs)} duplicate pairs. Returning top {len(results)}.")
        
        return ScanInternalResponse(
            total_scanned=len(all_products),
            pairs_found=len(duplicate_pairs),
            results=results,
            processing_time=round(processing_time, 3),
            fallback_active=fallback_active,
            system_warnings=system_warnings
        )
        
    except Exception as e:
        logger.exception("Failed to scan internal duplicates")
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")
