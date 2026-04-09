import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from supabase import create_client, Client
from tqdm import tqdm

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Import local modules
sys.path.append(os.getcwd())
from advanced_models import SentenceTransformerModel
from fresh_implementations import ThaiTextProcessor

# Configuration
SUPABASE_URL = "http://localhost:54331"
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
INPUT_FILE = "input/รายการสินค้าพร้อมหมวดหมู่_AI.txt"

# Initialize Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class LessonImporter:
    def __init__(self):
        print("🔧 Initializing Lesson Importer...")
        self.processor = ThaiTextProcessor()
        self.model = SentenceTransformerModel(model_name="paraphrase-multilingual-MiniLM-L12-v2")
        self.db_nodes = {}
        
    def load_taxonomy(self):
        print("📚 Loading taxonomy mapping from Supabase...")
        res = supabase.table("taxonomy_nodes").select("id, name_th").execute()
        self.db_nodes = {node['name_th'].strip(): node['id'] for node in res.data}
        print(f"✅ Loaded {len(self.db_nodes)} categories for mapping.")

    def run_import(self):
        self.load_taxonomy()
        
        # 1. Read the Reference File
        df = pd.read_csv(INPUT_FILE, sep='\t', encoding='utf-16')
        
        # Mapping column indices based on previous analysis:
        # 1: itemid (sku)
        # 3: name_th
        # 4: price
        # 9: sub_category (lesson)
        
        sku_col = df.columns[1]
        name_col = df.columns[3]
        price_col = df.columns[4]
        sub_cat_col = df.columns[9]
        
        # Create Import Batch
        import_batch = supabase.table("imports").insert({
            "name": f"Lesson Data Import - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "file_name": os.path.basename(INPUT_FILE),
            "status": "processing",
            "total_records": len(df)
        }).execute().data[0]
        batch_id = import_batch['id']
        
        print(f"🚀 Importing {len(df)} lessons into batch: {batch_id}")
        
        success_count = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Lessons"):
            try:
                sku = str(row[sku_col]).strip()
                name_th = str(row[name_col]).strip()
                price = float(row[price_col]) if not pd.isna(row[price_col]) else 0.0
                sub_cat_name = str(row[sub_cat_col]).strip()
                
                # Map to Category ID
                cat_id = self.db_nodes.get(sub_cat_name)
                if not cat_id:
                    # Try fuzzy match for slashes
                    fuzzy_name = sub_cat_name.replace(' / ', '/').replace(' /', '/').replace('/ ', '/')
                    for db_name, db_id in self.db_nodes.items():
                        if db_name.replace(' / ', '/').replace(' /', '/').replace('/ ', '/') == fuzzy_name:
                            cat_id = db_id
                            break
                
                if not cat_id:
                    print(f"⚠️ Category not found for product {name_th}: {sub_cat_name}")
                    continue
                
                # Process & Embed
                clean_name = self.processor.process(name_th)
                embedding = self.model.encode([clean_name])[0]
                
                # Insert Product as APPROVED lesson
                supabase.table("products").insert({
                    "name_th": name_th,
                    "sku": sku,
                    "price": price,
                    "category_id": cat_id,
                    "embedding": embedding.tolist(),
                    "confidence_score": 1.0, # Lessons are 100% correct
                    "status": "approved",
                    "import_batch_id": batch_id,
                    "metadata": {
                        "clean_name": clean_name,
                        "source": "lesson_file",
                        "original_sub_category": sub_cat_name
                    }
                }).execute()
                
                success_count += 1
            except Exception as e:
                print(f"❌ Error processing SKU {sku if 'sku' in locals() else 'unknown'}: {e}")
        
        # Complete Batch
        supabase.table("imports").update({
            "status": "completed",
            "success_records": success_count,
            "completed_at": datetime.now().isoformat()
        }).eq("id", batch_id).execute()
        
        print(f"✅ Lesson Import finished! Successfully imported {success_count} / {len(df)} products.")

if __name__ == "__main__":
    importer = LessonImporter()
    importer.run_import()
