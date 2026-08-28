from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import time
import os
import json
import logging
import numpy as np
from typing import List, Optional, Dict

from src.api.models import (
    ScanInternalRequest, ScanInternalResponse, InternalDuplicatePair,
    FileDeduplicationRequest, FileDeduplicationResponse, ProductMatchResult, FileDeduplicationStats, ImportDeduplicationRequest
)
from src.api.dependencies import get_taxonomy_service, get_ml_learning_system, get_supabase_client, initialize_embedding_model
from src.services.human_feedback_system import ProductComparison, FeedbackType
import csv
import io

# Global in-memory storage for asynchronous scan tasks
SCAN_TASKS: Dict[str, dict] = {}

# ป้ายที่แปลว่า "สินค้าคู่นี้คือตัวเดียวกัน (มีในสตอกแล้ว)"
#
# ระวังคำศัพท์สองชุดที่ชนกันในระบบนี้:
#   1. FeedbackType.SIMILAR = "สินค้าคล้าย แต่**ไม่ซ้ำ**" (คนละสินค้า เช่น กระชอนพลาสติก vs กระชอนเลส)
#   2. ฟิลด์ mlPrediction ที่ API ส่งออก ใช้คำว่า "similar" แปลว่า "เจอคู่ที่ตรงกัน ควรให้คนดู"
# คนละความหมายกันคนละชุด — ตรงนี้ยึดความหมายชุดที่ 1
# มีแต่ DUPLICATE เท่านั้นที่แปลว่า "สินค้าตัวนี้มีในสตอกแล้ว ไม่ต้องนำเข้า"
#
# บั๊กเดิม: โค้ดเทียบ pred_label กับ FeedbackType.SIMILAR.value ซึ่งโมเดลไม่เคยทำนายออกมาเลย
# (เทรนจาก similarity_matches ที่มีแค่ 'duplicate'/'different' — ดู
# ml_feedback_learning._fetch_training_data) เงื่อนไขจึงเป็นเท็จเสมอ ทุกคู่ถูกรายงานว่า
# "different" หมด รวมถึงคู่ที่ต่างกันแค่ช่องว่าง เช่น
# "แขวนเสื้อลวด+หนีบ 99 SM" vs "แขวนเสื้อลวด + หนีบ 99 SM" (ความคล้าย 0.96)
MATCH_LABELS = frozenset({FeedbackType.DUPLICATE.value})


def is_match_label(pred_label) -> bool:
    """โมเดลทำนายว่าคู่นี้เป็นสินค้าตัวเดียวกันหรือไม่

    ใช้ตัดสินว่าสินค้าใหม่ที่กำลังตรวจ "มีในสตอกแล้ว" หรือ "ยังไม่เคยมี"
    """
    return str(pred_label) in MATCH_LABELS

class ScanStartResponse(BaseModel):
    task_id: str
    status: str
    message: str

class ScanTaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: int
    message: str
    result: Optional[dict] = None
    error: Optional[str] = None

def parse_csv(file_bytes: bytes) -> List[str]:
    """Parse product names from CSV file bytes dynamically."""
    try:
        text = file_bytes.decode('utf-8-sig')
    except UnicodeDecodeError:
        text = file_bytes.decode('cp1252', errors='replace')
        
    f = io.StringIO(text)
    reader = csv.reader(f)
    try:
        rows = list(reader)
    except Exception:
        # Fallback split-line parser in case of parsing errors
        rows = [line.split(',') for line in text.split('\n') if line.strip()]
        
    if not rows:
        return []
    
    header = rows[0]
    name_col_idx = 0
    common_names = ["name_th", "name", "title", "product", "product_name", "ชื่อสินค้า", "ชื่อ"]
    for i, col in enumerate(header):
        col_clean = col.strip().lower()
        if col_clean in common_names or any(cn in col_clean for cn in common_names):
            name_col_idx = i
            break
            
    products = []
    for row in rows[1:]:
        if not row:
            continue
        if len(row) > name_col_idx:
            val = row[name_col_idx].strip().replace('"', '')
            if val:
                products.append(val)
        else:
            val = row[0].strip().replace('"', '') if row else ""
            if val:
                products.append(val)
                
    return products


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/match", tags=["Matching"])

def run_scan_task_background(task_id: str, threshold: float, limit: int):
    """Execute scan_internal_duplicates logic in the background and update SCAN_TASKS."""
    start_time = time.time()
    SCAN_TASKS[task_id] = {
        "status": "processing",
        "progress": 5,
        "message": "กำลังเตรียมตัวเชื่อมต่อฐานข้อมูล...",
        "result": None,
        "error": None
    }
    try:
        # Get Taxonomy Service to obtain Supabase Client
        SCAN_TASKS[task_id]["message"] = "กำลังเชื่อมต่อกับ Supabase..."
        SCAN_TASKS[task_id]["progress"] = 10
        service = get_taxonomy_service()
        if not service or not service.supabase:
            raise Exception("Supabase connection unavailable")
            
        supabase = service.supabase
        ml_system = get_ml_learning_system()
        
        # 1. Fetch Taxonomy Category Names Mapping
        SCAN_TASKS[task_id]["message"] = "กำลังโหลดข้อมูลหมวดหมู่และประวัติรีวิว..."
        SCAN_TASKS[task_id]["progress"] = 15
        cat_res = supabase.table("taxonomy_nodes").select("id, name_th").execute()
        category_mapping = {cat["id"]: cat["name_th"] for cat in cat_res.data}
        
        # 1.5 Fetch reviewed matches to prevent showing them again
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
        
        # 2. Fetch all Approved Products in pages
        SCAN_TASKS[task_id]["message"] = "กำลังดาวน์โหลดรายการสินค้าที่ได้รับการอนุมัติ..."
        SCAN_TASKS[task_id]["progress"] = 25
        all_products = []
        page_size = 1000
        offset = 0
        
        while True:
            res = supabase.table("products")\
                          .select("id, name_th, sku, category_id")\
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
                
        # 3. Clean product names using ThaiTextProcessor
        SCAN_TASKS[task_id]["message"] = "กำลังทำความสะอาดชื่อสินค้าภาษาไทย..."
        SCAN_TASKS[task_id]["progress"] = 40
        from src.core.fresh_implementations import ThaiTextProcessor
        processor = ThaiTextProcessor()
        
        valid_products = [p for p in all_products if p.get('name_th')]
        
        if len(valid_products) < 2:
            SCAN_TASKS[task_id] = {
                "status": "completed",
                "progress": 100,
                "message": "สแกนเสร็จสิ้น (คลังสินค้ามีรายการน้อยกว่า 2 รายการ)",
                "result": {
                    "total_scanned": len(all_products),
                    "pairs_found": 0,
                    "results": [],
                    "processing_time": round(time.time() - start_time, 3),
                    "fallback_active": False,
                    "system_warnings": []
                },
                "error": None
            }
            return
            
        # 4. Extract data, clean names, and generate fresh embeddings dynamically
        SCAN_TASKS[task_id]["message"] = f"กำลังคำนวณเวกเตอร์เอ็มเบดดิงสดให้กับสินค้า {len(valid_products)} รายการ..."
        SCAN_TASKS[task_id]["progress"] = 55
        ids = [p['id'] for p in valid_products]
        names = [p['name_th'] for p in valid_products]
        skus = [p['sku'] or "N/A" for p in valid_products]
        cat_ids = [p['category_id'] or "" for p in valid_products]
        
        cleaned_names = [processor.process(name) for name in names]
        
        # Generate fresh embeddings using the initialized model
        embedding_model = initialize_embedding_model()
        emb_matrix = embedding_model.encode(cleaned_names)
        emb_matrix = np.array(emb_matrix, dtype=np.float32)
        
        # 5. Normalize embeddings for fast Cosine Similarity via dot product
        SCAN_TASKS[task_id]["message"] = "ปัญญาประดิษฐ์กำลังวิเคราะห์และสแกนกลุ่มสินค้าซ้ำซ้อน..."
        SCAN_TASKS[task_id]["progress"] = 75
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_emb = emb_matrix / norms
        
        # Compute similarity matrix
        similarity_matrix = np.dot(normalized_emb, normalized_emb.T)
        
        # VECTORIZED FILTERING:
        triu_mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
        match_mask = (similarity_matrix >= threshold) & triu_mask
        i_indices, j_indices = np.where(match_mask)
        
        # 6. Process candidate pairs and apply batch ML inference
        SCAN_TASKS[task_id]["message"] = f"กำลังสแกนเปรียบเทียบคู่สินค้าก้ำกึ่ง {len(i_indices)} คู่ ด้วย Machine Learning..."
        SCAN_TASKS[task_id]["progress"] = 85
        duplicate_pairs = []
        fallback_active = False
        system_warnings = []
        
        # --- STAGE 2: Batch ML Inference ---
        if ml_system and ml_system.model.is_trained and len(i_indices) > 0:
            try:
                cleaned_names_ml = [ml_system.text_processor.process(name) for name in names]
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
                        product1_cleaned=cleaned_names_ml[i],
                        product2_cleaned=cleaned_names_ml[j],
                        similarity_score=sim,
                        confidence_score=0.8,
                        ml_prediction=FeedbackType.SIMILAR
                    )
                    comparisons.append(comp)
                    candidate_indices.append((i, j, sim))
                    
                if comparisons:
                    # Extract features for all comparisons in one vectorized call
                    X = ml_system.model.feature_extractor.extract_features(comparisons)
                    
                    # Run predictions in batch
                    predictions = ml_system.model.model.predict(X)
                    probabilities = ml_system.model.model.predict_proba(X)
                    predicted_labels = ml_system.model.label_encoder.inverse_transform(predictions)
                    
                    for idx, (i, j, sim) in enumerate(candidate_indices):
                        pred_label = predicted_labels[idx]
                        conf = float(max(probabilities[idx]))
                        
                        cat_a = category_mapping.get(cat_ids[i], "Unmapped")
                        cat_b = category_mapping.get(cat_ids[j], "Unmapped")
                        
                        ml_pred_str = "similar" if is_match_label(pred_label) else "different"
                        ml_label_th = "อาจซ้ำกัน" if ml_pred_str == "similar" else "อาจแตกต่าง"
                        reason_str = f"ความคล้ายเวกเตอร์ {sim*100:.1f}% | ML ประเมิน: {ml_label_th} (มั่นใจ {conf*100:.1f}%)"
                        
                        duplicate_pairs.append({
                            "id_a": ids[i],
                            "name_a": names[i],
                            "sku_a": skus[i],
                            "category_a": cat_a,
                            "category_id_a": cat_ids[i],
                            "id_b": ids[j],
                            "name_b": names[j],
                            "sku_b": skus[j],
                            "category_b": cat_b,
                            "category_id_b": cat_ids[j],
                            "similarity": round(sim, 4),
                            "ml_prediction": ml_pred_str,
                            "confidence": round(conf, 4),
                            "reason": reason_str
                        })
            except Exception as e:
                logger.exception("Batch ML Inference failed. Falling back to non-ML scan.")
                fallback_active = True
                system_warnings.append(f"ระบบ AI ขัดข้อง: {str(e)}")
                # Fallback in case batch ML prediction fails: process without ML filtering
                for idx in range(len(i_indices)):
                    i, j = int(i_indices[idx]), int(j_indices[idx])
                    sim = float(similarity_matrix[i, j])
                    
                    pair_key = tuple(sorted([ids[i], ids[j]]))
                    if pair_key in reviewed_pairs:
                        continue
                        
                    cat_a = category_mapping.get(cat_ids[i], "Unmapped")
                    cat_b = category_mapping.get(cat_ids[j], "Unmapped")
                    
                    reason_str = f"ความคล้ายเวกเตอร์ {sim*100:.1f}% (ระบบสำรอง - ไม่มีข้อมูล ML)"
                    
                    duplicate_pairs.append({
                        "id_a": ids[i],
                        "name_a": names[i],
                        "sku_a": skus[i],
                        "category_a": cat_a,
                        "category_id_a": cat_ids[i],
                        "id_b": ids[j],
                        "name_b": names[j],
                        "sku_b": skus[j],
                        "category_b": cat_b,
                        "category_id_b": cat_ids[j],
                        "similarity": round(sim, 4),
                        "ml_prediction": "similar",
                        "confidence": round(sim, 4),
                        "reason": reason_str
                    })
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
                
                reason_str = f"ความคล้ายเวกเตอร์ {sim*100:.1f}% (ระบบสำรอง - ไม่มีข้อมูล ML)"
                
                duplicate_pairs.append({
                    "id_a": ids[i],
                    "name_a": names[i],
                    "sku_a": skus[i],
                    "category_a": cat_a,
                    "category_id_a": cat_ids[i],
                    "id_b": ids[j],
                    "name_b": names[j],
                    "sku_b": skus[j],
                    "category_b": cat_b,
                    "category_id_b": cat_ids[j],
                    "similarity": round(sim, 4),
                    "ml_prediction": "similar",
                    "confidence": round(sim, 4),
                    "reason": reason_str
                })
            
        # Sort pairs by similarity descending
        duplicate_pairs.sort(key=lambda x: x["similarity"], reverse=True)
        results = duplicate_pairs[:limit]
        processing_time = time.time() - start_time
        
        SCAN_TASKS[task_id] = {
            "status": "completed",
            "progress": 100,
            "message": "สแกนตรวจสอบเสร็จสิ้นเรียบร้อยแล้ว!",
            "result": {
                "total_scanned": len(all_products),
                "pairs_found": len(duplicate_pairs),
                "results": results,
                "processing_time": round(processing_time, 3),
                "fallback_active": fallback_active,
                "system_warnings": system_warnings
            },
            "error": None
        }
        
    except Exception as e:
        logger.exception(f"Background scan task {task_id} failed")
        SCAN_TASKS[task_id] = {
            "status": "failed",
            "progress": 100,
            "message": "เกิดข้อผิดพลาดในการประมวลผล",
            "result": None,
            "error": str(e)
        }

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/match", tags=["Matching"])

@router.post("/scan-internal", response_model=ScanStartResponse)
async def scan_internal_duplicates(request: ScanInternalRequest, background_tasks: BackgroundTasks):
    """
    Start an asynchronous background task to scan approved products for duplicates using pairwise cosine similarity.
    """
    import uuid
    task_id = str(uuid.uuid4())
    SCAN_TASKS[task_id] = {
        "status": "pending",
        "progress": 0,
        "message": "กำลังยืนยันคิวงานสแกน...",
        "result": None,
        "error": None
    }
    background_tasks.add_task(run_scan_task_background, task_id, request.threshold, request.limit)
    return ScanStartResponse(
        task_id=task_id,
        status="pending",
        message="เริ่มสแกนตรวจสอบในเบื้องหลังเรียบร้อยแล้ว"
    )

@router.get("/scan-internal/status/{task_id}", response_model=ScanTaskStatusResponse)
async def get_internal_scan_status(task_id: str):
    """
    Get the status, progress, or results of a background scan task.
    """
    task = SCAN_TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="ไม่พบรหัสงานสแกนที่ระบุ")
    return ScanTaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        message=task["message"],
        result=task["result"],
        error=task["error"]
    )


@router.post("/file-dedup", response_model=FileDeduplicationResponse)
async def deduplicate_files(request: FileDeduplicationRequest):
    """
    Deduplicate uploaded CSV product files using hybrid vectorized cosine similarity
    coupled with ML classification batch inference.
    """
    start_time = time.time()
    try:
        # 1. Get Supabase client and dependencies
        supabase = get_supabase_client()
        if not supabase:
            raise HTTPException(status_code=503, detail="Supabase connection unavailable")
            
        ml_system = get_ml_learning_system()
        embedding_model = initialize_embedding_model()
        
        # 2. Download files from Supabase Storage
        logger.info(f"Downloading old products file: {request.old_products_path}")
        try:
            old_bytes = supabase.storage.from_("uploads").download(request.old_products_path)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Failed to download old products file: {str(e)}")
            
        logger.info(f"Downloading new products file: {request.new_products_path}")
        try:
            new_bytes = supabase.storage.from_("uploads").download(request.new_products_path)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Failed to download new products file: {str(e)}")
            
        # 3. Parse products
        old_products = parse_csv(old_bytes)
        new_products = parse_csv(new_bytes)
        
        logger.info(f"Parsed {len(new_products)} new products vs {len(old_products)} old products.")
        
        if not old_products or not new_products:
            # Empty file handling
            return FileDeduplicationResponse(
                success=True,
                results=[],
                uniqueProducts=new_products,
                stats=FileDeduplicationStats(
                    totalNewProducts=len(new_products),
                    totalOldProducts=len(old_products),
                    needsReview=0,
                    autoApproved=len(new_products),
                    excludedDuplicates=0,
                    processingTime=round(time.time() - start_time, 3)
                ),
                summary="ไม่มีผลิตภัณฑ์ใหม่หรือผลิตภัณฑ์อ้างอิงให้ประมวลผล"
            )
            
        # 4. Generate Embeddings
        logger.info("Generating embeddings for file deduplication...")
        from src.core.fresh_implementations import ThaiTextProcessor
        processor = ThaiTextProcessor()
        cleaned_old_products = [processor.process(name) for name in old_products]
        cleaned_new_products = [processor.process(name) for name in new_products]
        old_embs = embedding_model.encode(cleaned_old_products)
        new_embs = embedding_model.encode(cleaned_new_products)
        
        # 5. Compute Vector similarity matrix (Cosine)
        # shape: (N, M)
        norms_new = np.linalg.norm(new_embs, axis=1, keepdims=True)
        norms_new[norms_new == 0] = 1.0
        normalized_new = new_embs / norms_new

        norms_old = np.linalg.norm(old_embs, axis=1, keepdims=True)
        norms_old[norms_old == 0] = 1.0
        normalized_old = old_embs / norms_old

        similarity_matrix = np.dot(normalized_new, normalized_old.T)
        
        # 6. Find best candidate matches
        candidates = []
        unique_products = []
        
        for i, new_product in enumerate(new_products):
            j = int(np.argmax(similarity_matrix[i]))
            sim = float(similarity_matrix[i, j])
            
            if sim >= 0.95:
                # Highly similar - automatic duplicate, we exclude it from unique products
                continue
            elif sim >= request.threshold:
                candidates.append((i, j, sim))
            else:
                unique_products.append(new_product)
                
        # 7. Apply ML Inference on candidates
        matches = []
        
        if ml_system and ml_system.model.is_trained and candidates:
            try:
                logger.info(f"Running ML inference on {len(candidates)} candidates...")
                cleaned_new_names = [ml_system.text_processor.process(new_products[i]) for i, _, _ in candidates]
                cleaned_old_names = [ml_system.text_processor.process(old_products[j]) for _, j, _ in candidates]
                
                comparisons = []
                for idx, (i, j, sim) in enumerate(candidates):
                    comp = ProductComparison(
                        id="inference",
                        product1=new_products[i],
                        product2=old_products[j],
                        product1_cleaned=cleaned_new_names[idx],
                        product2_cleaned=cleaned_old_names[idx],
                        similarity_score=sim,
                        confidence_score=0.8,
                        ml_prediction=FeedbackType.SIMILAR
                    )
                    comparisons.append(comp)
                    
                X = ml_system.model.feature_extractor.extract_features(comparisons)
                predictions = ml_system.model.model.predict(X)
                probabilities = ml_system.model.model.predict_proba(X)
                predicted_labels = ml_system.model.label_encoder.inverse_transform(predictions)
                
                for idx, (i, j, sim) in enumerate(candidates):
                    pred_label = predicted_labels[idx]
                    conf = float(max(probabilities[idx]))
                    
                    # If ML is confident they are DIFFERENT, treat it as unique and skip listing
                    if pred_label == FeedbackType.DIFFERENT.value and conf > 0.6:
                        unique_products.append(new_products[i])
                        continue
                        
                    # Otherwise, add to review zone
                    # Calculate reasonable confidence
                    matched = is_match_label(pred_label)
                    confidence = float(min(0.95, sim + 0.1)) if matched else float(sim)
                    ml_pred = "similar" if matched else "different"
                    
                    matches.append(ProductMatchResult(
                        id=f"review_{i + 1}",
                        newProduct=new_products[i],
                        oldProduct=old_products[j],
                        similarity=round(sim, 4),
                        confidence=round(confidence, 4),
                        mlPrediction=ml_pred,
                        status="pending",
                        reason=f"ความคล้าย {sim*100:.1f}% - {'อาจซ้ำกัน' if ml_pred == 'similar' else 'อาจแตกต่าง'}"
                    ))
            except Exception as ml_err:
                logger.exception(f"ML inference failed in file deduplication: {ml_err}")
                # Fallback to rules if ML fails
                for i, j, sim in candidates:
                    confidence = float(min(0.95, sim + 0.1))
                    ml_pred = "similar" if sim > 0.8 else "different"
                    matches.append(ProductMatchResult(
                        id=f"review_{i + 1}",
                        newProduct=new_products[i],
                        oldProduct=old_products[j],
                        similarity=round(sim, 4),
                        confidence=round(confidence, 4),
                        mlPrediction=ml_pred,
                        status="pending",
                        reason=f"ความคล้าย {sim*100:.1f}% - {'อาจซ้ำกัน' if ml_pred == 'similar' else 'อาจแตกต่าง'}"
                    ))
        else:
            # Fallback when ML Model is not trained / not initialized
            logger.info("ML system not active or trained. Using default Cosine similarity thresholds.")
            for i, j, sim in candidates:
                confidence = float(min(0.95, sim + 0.1))
                ml_pred = "similar" if sim > 0.8 else "different"
                matches.append(ProductMatchResult(
                    id=f"review_{i + 1}",
                    newProduct=new_products[i],
                    oldProduct=old_products[j],
                    similarity=round(sim, 4),
                    confidence=round(confidence, 4),
                    mlPrediction=ml_pred,
                    status="pending",
                    reason=f"ความคล้าย {sim*100:.1f}% - {'อาจซ้ำกัน' if ml_pred == 'similar' else 'อาจแตกต่าง'}"
                ))
                
        processing_time = time.time() - start_time
        
        # Calculate stats
        total_new = len(new_products)
        total_old = len(old_products)
        needs_review = len(matches)
        auto_approved = len(unique_products)
        excluded_duplicates = total_new - auto_approved - needs_review
        
        stats = FileDeduplicationStats(
            totalNewProducts=total_new,
            totalOldProducts=total_old,
            needsReview=needs_review,
            autoApproved=auto_approved,
            excludedDuplicates=excluded_duplicates,
            processingTime=round(processing_time, 3)
        )
        
        summary = f"จาก {total_new} สินค้าใหม่: {auto_approved} ไม่ซ้ำ, {needs_review} ต้องตรวจสอบ, {excluded_duplicates} ซ้ำมาก (ตัดออก)"
        
        return FileDeduplicationResponse(
            success=True,
            results=matches,
            uniqueProducts=unique_products,
            stats=stats,
            summary=summary
        )
    except Exception as e:
        logger.exception("File deduplication failed")
        raise HTTPException(status_code=500, detail=f"File deduplication failed: {str(e)}")


@router.post("/import-dedup", response_model=List[ProductMatchResult])
async def deduplicate_imports(request: ImportDeduplicationRequest):
    """
    Deduplicate a list of imported product names against the existing approved database products
    using hybrid vectorized cosine similarity coupled with ML classification batch inference.
    """
    try:
        # 1. Get Supabase client and dependencies
        supabase = get_supabase_client()
        if not supabase:
            raise HTTPException(status_code=503, detail="Supabase connection unavailable")
            
        ml_system = get_ml_learning_system()
        embedding_model = initialize_embedding_model()
        
        input_products = request.products
        if not input_products:
            return []
            
        # 2. Fetch approved products from database
        logger.info("Fetching approved products for import deduplication...")
        res = supabase.table("products")\
                      .select("id, name_th, embedding, price")\
                      .eq("status", "approved")\
                      .execute()
        
        db_products = res.data or []
        if not db_products:
            return []
            
        # 3. Extract database embeddings
        db_names = [p['name_th'] for p in db_products]
        db_ids = [p['id'] for p in db_products]
        
        db_embs = []
        valid_indices = []
        for idx, p in enumerate(db_products):
            emb = p.get('embedding')
            if emb:
                if isinstance(emb, str):
                    import json
                    emb = json.loads(emb)
                db_embs.append(emb)
                valid_indices.append(idx)
                
        if not db_embs:
            return []
            
        db_embs = np.array(db_embs, dtype=np.float32)
        
        # 4. Generate Embeddings for input products
        logger.info(f"Generating embeddings for {len(input_products)} imported products...")
        from src.core.fresh_implementations import ThaiTextProcessor
        processor = ThaiTextProcessor()
        cleaned_input_products = [processor.process(name) for name in input_products]
        input_embs = embedding_model.encode(cleaned_input_products)
        
        # 5. Compute Cosine similarity matrix (Cosine)
        # shapes: input_embs is (N, D), db_embs is (M, D)
        norms_in = np.linalg.norm(input_embs, axis=1, keepdims=True)
        norms_in[norms_in == 0] = 1.0
        normalized_in = input_embs / norms_in

        norms_db = np.linalg.norm(db_embs, axis=1, keepdims=True)
        norms_db[norms_db == 0] = 1.0
        normalized_db = db_embs / norms_db

        similarity_matrix = np.dot(normalized_in, normalized_db.T)
        
        # 6. Find best candidate matches above threshold
        candidates = []
        for i, input_product in enumerate(input_products):
            j_valid = int(np.argmax(similarity_matrix[i]))
            sim = float(similarity_matrix[i, j_valid])
            
            if sim >= request.threshold:
                # Map back to original db_products index
                j_orig = valid_indices[j_valid]
                candidates.append((i, j_orig, sim))
                
        # 7. Apply ML Inference on candidates
        matches = []
        
        if ml_system and ml_system.model.is_trained and candidates:
            try:
                logger.info(f"Running ML inference on {len(candidates)} candidates...")
                cleaned_new_names = [ml_system.text_processor.process(input_products[i]) for i, _, _ in candidates]
                cleaned_old_names = [ml_system.text_processor.process(db_products[j]['name_th']) for _, j, _ in candidates]
                
                comparisons = []
                for idx, (i, j, sim) in enumerate(candidates):
                    comp = ProductComparison(
                        id="inference",
                        product1=input_products[i],
                        product2=db_products[j]['name_th'],
                        product1_cleaned=cleaned_new_names[idx],
                        product2_cleaned=cleaned_old_names[idx],
                        similarity_score=sim,
                        confidence_score=0.8,
                        ml_prediction=FeedbackType.SIMILAR
                    )
                    comparisons.append(comp)
                    
                X = ml_system.model.feature_extractor.extract_features(comparisons)
                predictions = ml_system.model.model.predict(X)
                probabilities = ml_system.model.model.predict_proba(X)
                predicted_labels = ml_system.model.label_encoder.inverse_transform(predictions)
                
                for idx, (i, j, sim) in enumerate(candidates):
                    pred_label = predicted_labels[idx]
                    conf = float(max(probabilities[idx]))
                    
                    # If ML says they are different, don't flag as duplicate
                    if pred_label == FeedbackType.DIFFERENT.value and conf > 0.6:
                        continue
                        
                    matched = is_match_label(pred_label)
                    confidence = float(min(0.95, sim + 0.1)) if matched else float(sim)
                    ml_pred = "similar" if matched else "different"
                    
                    matches.append(ProductMatchResult(
                        id=f"review_{i + 1}",
                        newProduct=input_products[i],
                        oldProduct=db_products[j]['name_th'],
                        oldProductId=db_products[j].get('id'),
                        oldPrice=db_products[j].get('price'),
                        similarity=round(sim, 4),
                        confidence=round(confidence, 4),
                        mlPrediction=ml_pred,
                        status="pending",
                        reason=f"ความคล้าย {sim*100:.1f}% - {'อาจซ้ำกัน' if ml_pred == 'similar' else 'อาจแตกต่าง'}"
                    ))
            except Exception as ml_err:
                logger.exception(f"ML inference failed in batch import deduplication: {ml_err}")
                # Fallback to rules if ML fails
                for i, j, sim in candidates:
                    confidence = float(min(0.95, sim + 0.1))
                    ml_pred = "similar" if sim > 0.8 else "different"
                    matches.append(ProductMatchResult(
                        id=f"review_{i + 1}",
                        newProduct=input_products[i],
                        oldProduct=db_products[j]['name_th'],
                        oldProductId=db_products[j].get('id'),
                        oldPrice=db_products[j].get('price'),
                        similarity=round(sim, 4),
                        confidence=round(confidence, 4),
                        mlPrediction=ml_pred,
                        status="pending",
                        reason=f"ความคล้าย {sim*100:.1f}% - {'อาจซ้ำกัน' if ml_pred == 'similar' else 'อาจแตกต่าง'}"
                    ))
        else:
            # Fallback when ML Model is not trained / not initialized
            logger.info("ML system not active or trained for batch import. Using default Cosine similarity thresholds.")
            for i, j, sim in candidates:
                confidence = float(min(0.95, sim + 0.1))
                ml_pred = "similar" if sim > 0.8 else "different"
                matches.append(ProductMatchResult(
                    id=f"review_{i + 1}",
                    newProduct=input_products[i],
                    oldProduct=db_products[j]['name_th'],
                    oldProductId=db_products[j].get('id'),
                    oldPrice=db_products[j].get('price'),
                    similarity=round(sim, 4),
                    confidence=round(confidence, 4),
                    mlPrediction=ml_pred,
                    status="pending",
                    reason=f"ความคล้าย {sim*100:.1f}% - {'อาจซ้ำกัน' if ml_pred == 'similar' else 'อาจแตกต่าง'}"
                ))
                
        return matches
        
    except Exception as e:
        logger.exception("Import deduplication failed")
        raise HTTPException(status_code=500, detail=f"Import deduplication failed: {str(e)}")


