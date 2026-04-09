import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Any

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Import local modules
sys.path.append(os.getcwd())
from fresh_implementations import ThaiTextProcessor, ComponentFactory
from advanced_models import SentenceTransformerModel

class StandardizedImportPipeline:
    """
    User-Controlled Standardized Pipeline: Clean -> Dedup -> Classify
    """
    def __init__(self, 
                 dedup_threshold: float = 0.95, 
                 classify_threshold: float = 0.80):
        print(f"🔧 Initializing Pipeline (Dedup: {dedup_threshold}, Classify: {classify_threshold})...")
        self.processor = ThaiTextProcessor(normalize_numbers=True, normalize_thai_chars=True)
        self.model = SentenceTransformerModel(model_name="paraphrase-multilingual-MiniLM-L12-v2")
        self.dedup_threshold = dedup_threshold
        self.classify_threshold = classify_threshold
        
        # สำหรับเก็บข้อมูล Taxonomy
        self.category_embeddings = {}
        self.category_names = {}
        self.keyword_rules = []
        
    def load_taxonomy(self, supabase_client):
        """โหลดข้อมูลหมวดหมู่จาก Supabase"""
        print("📚 Loading Taxonomy and Rules...")
        res = supabase_client.table("taxonomy_nodes").select("id, name_th, embedding").execute()
        for node in res.data:
            self.category_names[node['id']] = node['name_th']
            if node.get('embedding'):
                emb = node['embedding']
                if isinstance(emb, str): emb = json.loads(emb)
                self.category_embeddings[node['id']] = np.array(emb)
        
        res = supabase_client.table("keyword_rules").select("*").execute()
        self.keyword_rules = res.data
        print(f"✅ Loaded {len(self.category_names)} categories and {len(self.keyword_rules)} rules.")

    def run(self, input_file: str, supabase_client=None):
        # 1. Load Data
        print(f"📂 Loading input: {input_file}")
        try:
            df = pd.read_csv(input_file, encoding='utf-8')
        except:
            df = pd.read_csv(input_file, encoding='cp874')
        
        name_col = 'รายการ' if 'รายการ' in df.columns else df.columns[1]
        sku_col = 'รหัสสินค้า' if 'รหัสสินค้า' in df.columns else 'sku'
        
        # 2. STEP 1: CLEANING
        print("🧹 Step 1: Cleaning names...")
        df['clean_name'] = [self.processor.process(str(name)) for name in tqdm(df[name_col])]
        
        # 3. STEP 2: DEDUPLICATION
        print("🔍 Step 2: Deduplicating (User Controlled)...")
        clean_names = df['clean_name'].tolist()
        embeddings = self.model.encode(clean_names)
        
        is_processed = [False] * len(df)
        db_records = []
        
        for i in range(len(df)):
            if is_processed[i]: continue
            
            # This is the "Master" (Unique representative)
            is_processed[i] = True
            
            # Check for potential duplicates of this master
            duplicates = []
            for j in range(i + 1, len(df)):
                if is_processed[j]: continue
                
                sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                
                if sim >= self.dedup_threshold or (sim >= 0.85): # Within review range
                    is_processed[j] = True
                    # This is a candidate duplicate
                    duplicates.append({
                        "name_th": str(df.iloc[j][name_col]),
                        "sku": str(df.iloc[j][sku_col]) if sku_col in df.columns else f"D-{datetime.now().timestamp()}-{j}",
                        "status": "pending_review_dedup",
                        "embedding": embeddings[j].tolist(),
                        "metadata": {
                            "clean_name": df.iloc[j]['clean_name'],
                            "duplicate_of": str(df.iloc[i][name_col]),
                            "similarity_score": round(float(sim), 4)
                        }
                    })
            
            # Classify the "Master" product
            name = df.iloc[i][name_col]
            clean_name = df.iloc[i]['clean_name']
            emb = embeddings[i]
            
            best_cat = None
            best_conf = 0.0
            
            for rule in self.keyword_rules:
                for kw in rule['keywords']:
                    if self.processor.process(kw) in clean_name:
                        conf = rule.get('confidence_score', 0.8)
                        if conf > best_conf:
                            best_conf = conf
                            best_cat = rule['category_id']
            
            best_emb_cat = None
            best_emb_sim = 0.0
            for cat_id, cat_emb in self.category_embeddings.items():
                sim = np.dot(emb, cat_emb) / (np.linalg.norm(emb) * np.linalg.norm(cat_emb))
                if sim > best_emb_sim:
                    best_emb_sim = sim
                    best_emb_cat = cat_id
            
            final_cat_id = best_cat if best_cat else best_emb_cat
            final_conf = best_conf * 0.6 + best_emb_sim * 0.4 if best_cat else best_emb_sim
            
            # Define Master Status
            master_status = "approved" if final_conf >= self.classify_threshold else "pending_review_category"
            
            # Add Master to records
            db_records.append({
                "name_th": str(name),
                "sku": str(df.iloc[i][sku_col]) if sku_col in df.columns else f"U-{datetime.now().timestamp()}-{i}",
                "category_id": final_cat_id,
                "status": master_status,
                "embedding": emb.tolist(),
                "confidence_score": round(float(final_conf), 2),
                "metadata": {
                    "clean_name": clean_name,
                    "is_representative": True,
                    "suggested_category": self.category_names.get(final_cat_id, "Unknown")
                }
            })
            
            # Add Duplicates to records
            db_records.extend(duplicates)

        # 4. Step 4: Sync to Supabase (The Storage Room)
        if supabase_client and db_records:
            print(f"🚀 Syncing {len(db_records)} items to Supabase storage...")
            # Batch insert (Supabase limit is usually around 1000)
            chunk_size = 100
            for k in range(0, len(db_records), chunk_size):
                chunk = db_records[k:k+chunk_size]
                supabase_client.table("products").insert(chunk).execute()
            print("✅ Data successfully stored in 'products' table.")
        else:
            # Fallback to CSV if no DB client
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pd.DataFrame(db_records).to_csv(f"output/db_sync_preview_{timestamp}.csv", index=False, encoding='utf-8-sig')
            print(f"⚠️ No DB client. Preview saved to output/db_sync_preview_{timestamp}.csv")

        print(f"📊 Summary: Total={len(db_records)} records processed.")

if __name__ == "__main__":
    # สำหรับรันแบบ Standalone ต้องมี Supabase Client
    from supabase import create_client
    load_dotenv = lambda: None # Mock
    
    # ดึงค่าจาก ENV หรือใส่ตรงๆ สำหรับทดสอบ
    SUPABASE_URL = "http://localhost:54331"
    SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    pipeline = StandardizedImportPipeline()
    pipeline.load_taxonomy(supabase)
    
    # ทดสอบกับไฟล์ตัวอย่าง
    input_p = r"input/new_product/POS_เพิ่มสินค้า_20250727_063658_จากไฟล์สินค้าใหม่.csv"
    if os.path.exists(input_p):
        pipeline.run(input_p)
    else:
        print(f"⚠️ Input file not found: {input_p}")
