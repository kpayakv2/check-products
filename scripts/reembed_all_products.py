#!/usr/bin/env python3
"""
Re-generate Embeddings for All Approved Products (High-Performance Batch & Async Version)
========================================================================================
สคริปต์ล้างและแปลงเวกเตอร์ (Re-embedding) สินค้าที่อนุมัติแล้วทั้งหมดในฐานข้อมูล
ด้วยขุมพลังการประมวลผลเวกเตอร์แบบ Batch และบันทึกข้อมูลแบบ Async/Parallel ยิงตรงเข้าฐานข้อมูล
เพื่อความเร็วระดับสูง (เสร็จสิ้นภายในไม่กี่วินาที แทนที่จะใช้เวลาหลายนาที)
"""

import sys
import os
import json
import logging
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import httpx

# เซตการเข้ารหัสเอาต์พุตให้รองรับภาษาไทยบนคอมพิวเตอร์ Windows
try:
    sys.stdout.reconfigure(errors='replace')
    sys.stderr.reconfigure(errors='replace')
except AttributeError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# โหลด Environment Variables
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / "taxonomy-app" / ".env.local"
load_dotenv(dotenv_path=str(env_path))

# เพิ่ม root dir ใน sys.path
sys.path.append(str(BASE_DIR))

from src.api.dependencies import get_supabase_client, initialize_embedding_model
from src.core.fresh_implementations import ThaiTextProcessor

async def update_product_emb(client: httpx.AsyncClient, product_id: str, embedding: list, sem: asyncio.Semaphore, headers: dict) -> bool:
    """Async task to update a single product embedding in PostgreSQL via PostgREST."""
    async with sem:
        url = f"http://127.0.0.1:54331/rest/v1/products?id=eq.{product_id}"
        try:
            # PostgREST patch update
            res = await client.patch(url, json={"embedding": embedding}, headers=headers)
            if res.status_code not in (200, 201, 204):
                logger.error(f"⚠️ Failed to update product ID {product_id}: {res.status_code} {res.text}")
                return False
            return True
        except Exception as e:
            logger.error(f"⚠️ Exception updating product ID {product_id}: {e}")
            return False

async def main_async():
    logger.info("🚀 Starting High-Performance POS Products Re-embedding Process...")
    
    # 1. Initialize Supabase Client to verify credentials
    supabase = get_supabase_client()
    if not supabase:
        logger.error("❌ Failed to initialize Supabase client. Check taxonomy-app/.env.local credentials.")
        return 1
        
    # Get supabase credentials for HTTP PostgREST API
    supabase_url = "http://127.0.0.1:54331"
    service_role_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not service_role_key:
        logger.error("❌ SUPABASE_SERVICE_ROLE_KEY not found in environment.")
        return 1
        
    headers = {
        "apikey": service_role_key,
        "Authorization": f"Bearer {service_role_key}",
        "Content-Type": "application/json"
    }

    # 2. Initialize Local 384-dim Embedding Model
    logger.info("🧠 Initializing local SentenceTransformer (paraphrase-multilingual-MiniLM-L12-v2)...")
    try:
        model = initialize_embedding_model()
    except Exception as e:
        logger.error(f"❌ Failed to load local embedding model: {e}")
        return 1

    # 3. Initialize Deduplication-configured ThaiTextProcessor (Spaces = 0)
    processor = ThaiTextProcessor(remove_all_spaces=True)
    logger.info("🌐 ThaiTextProcessor initialized with remove_all_spaces=True & unit protection!")

    # 4. Fetch all Approved Products from Database
    logger.info("🔍 Fetching approved products from Supabase database...")
    all_products = []
    page_size = 1000
    offset = 0
    
    while True:
        try:
            res = supabase.table("products")\
                          .select("id, name_th")\
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
        except Exception as e:
            logger.error(f"❌ Failed to fetch database products batch: {e}")
            return 1

    logger.info(f"📊 Found {len(all_products)} approved products in PostgreSQL.")
    if not all_products:
        logger.info("✅ No products to re-embed!")
        return 0

    # 5. Preprocess names in batch
    logger.info("🌐 Cleaning and preparing product names...")
    cleaned_names = []
    product_ids = []
    for p in all_products:
        cleaned_names.append(processor.process(p["name_th"]))
        product_ids.append(p["id"])

    # 6. Generate Embeddings using high-speed Batch encoding
    logger.info(f"🧠 Generating 384-dim vector embeddings for {len(cleaned_names)} products in batch...")
    try:
        embs = model.encode(cleaned_names)
        logger.info("✅ Finished generating embeddings! Preparing database update tasks...")
    except Exception as e:
        logger.error(f"❌ Failed to generate batch embeddings: {e}")
        return 1

    # 7. Async parallel database update using Semaphore for concurrency control
    logger.info("⚡ Executing parallel PostgreSQL update tasks...")
    sem = asyncio.Semaphore(50)  # Concurrency limit: 50 concurrent requests
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = []
        for idx, (p_id, emb) in enumerate(zip(product_ids, embs)):
            embedding_list = emb.tolist()
            tasks.append(
                update_product_emb(client, p_id, embedding_list, sem, headers)
            )
            
        # Run all updates concurrently and track progress
        results = await asyncio.gather(*tasks)
        
    success_count = sum(1 for r in results if r)
    logger.info(f"🎉 Successfully re-embedded {success_count}/{len(all_products)} approved products in PostgreSQL!")
    
    if success_count < len(all_products):
        logger.warning(f"⚠️ Warning: {len(all_products) - success_count} products failed to update.")
        
    return 0

def main():
    return asyncio.run(main_async())

if __name__ == "__main__":
    sys.exit(main())
