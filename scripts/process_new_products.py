import os
import sys
import json
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
INPUT_FILE = r"D:\product_checker\check-products\input\new_product\POS_เพิ่มสินค้า_20250727_063658_จากไฟล์สินค้าใหม่.csv"
OUTPUT_FILE = f"output/classified_new_products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# Initialize Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

class NewProductProcessor:
    def __init__(self):
        print("🔧 Initializing New Product Processor...")
        self.processor = ThaiTextProcessor()
        self.model = SentenceTransformerModel(model_name="paraphrase-multilingual-MiniLM-L12-v2")
        self.db_nodes = {}
        self.category_embeddings = {}
        self.keyword_rules = []
        self.reference_products = [] # The 3,103 lessons
        
    def load_metadata(self):
        print("📚 Loading metadata from Supabase...")
        # Load taxonomy
        res = supabase.table("taxonomy_nodes").select("id, name_th, keywords, embedding").execute()
        self.db_nodes = {node['id']: node['name_th'] for node in res.data}
        self.taxonomy_full = res.data
        for node in res.data:
            if node.get('embedding'):
                emb = node['embedding']
                if isinstance(emb, str):
                    emb = json.loads(emb)
                self.category_embeddings[node['id']] = np.array(emb)
        
        # Load keyword rules
        res = supabase.table("keyword_rules").select("*").execute()
        self.keyword_rules = res.data
        
        # Load reference products (lessons)
        print("📖 Loading reference lessons (3,103 products)...")
        # To avoid massive memory usage, we can limit or process in batches if needed,
        # but 3,103 is small enough for most systems.
        res = supabase.table("products").select("id, name_th, category_id, embedding").eq("status", "approved").execute()
        
        processed_lessons = []
        for lesson in res.data:
            if lesson.get('embedding'):
                emb = lesson['embedding']
                if isinstance(emb, str):
                    emb = json.loads(emb)
                lesson['embedding_arr'] = np.array(emb)
                processed_lessons.append(lesson)
        
        self.reference_products = processed_lessons
        print(f"✅ Loaded {len(self.taxonomy_full)} categories and {len(self.reference_products)} lessons.")

    def classify(self, name: str) -> dict:
        clean_name = self.processor.process(name)
        prod_emb = self.model.encode([clean_name])[0]
        
        # 1. Similarity with Lessons (The "Teacher" products)
        best_lesson_sim = 0.0
        best_lesson_cat = None
        
        for lesson in self.reference_products:
            lesson_emb = lesson['embedding_arr']
            sim = np.dot(prod_emb, lesson_emb) / (np.linalg.norm(prod_emb) * np.linalg.norm(lesson_emb))
            if sim > best_lesson_sim:
                best_lesson_sim = sim
                best_lesson_cat = lesson['category_id']
        
        # 2. Keyword Match
        best_kw_cat = None
        best_kw_conf = 0.0
        for rule in self.keyword_rules:
            for kw in rule['keywords']:
                if kw in clean_name:
                    conf = (rule.get('confidence_score', 0.8))
                    if conf > best_kw_conf:
                        best_kw_conf = conf
                        best_kw_cat = rule['category_id']
        
        # 3. Hybrid Calculation
        # Distance to Category Embeddings
        best_cat_emb_sim = 0.0
        best_cat_emb_id = None
        for cat_id, cat_emb in self.category_embeddings.items():
            sim = np.dot(prod_emb, cat_emb) / (np.linalg.norm(prod_emb) * np.linalg.norm(cat_emb))
            if sim > best_cat_emb_sim:
                best_cat_emb_sim = sim
                best_cat_emb_id = cat_id
        
        # Final Decision
        if best_lesson_sim > 0.94:
            final_cat_id = best_lesson_cat
            final_conf = best_lesson_sim
            method = "lesson_match"
        else:
            # Weight: Keyword 60%, Category Vector 40%
            if best_kw_cat:
                final_cat_id = best_kw_cat
                # If keyword matches, we combine it with category embedding similarity
                # but only if it's the SAME category
                if best_kw_cat == best_cat_emb_id:
                    final_conf = (best_kw_conf * 0.6 + best_cat_emb_sim * 0.4)
                else:
                    final_conf = best_kw_conf * 0.6 + (best_cat_emb_sim * 0.4)
                method = "hybrid"
            else:
                final_cat_id = best_cat_emb_id
                final_conf = best_cat_emb_sim
                method = "embedding"
            
        return {
            "category_id": final_cat_id,
            "category_name": self.db_nodes.get(final_cat_id, "Unknown"),
            "confidence": float(final_conf),
            "method": method,
            "lesson_sim": float(best_lesson_sim)
        }

    def run(self):
        self.load_metadata()
        
        # Read CSV
        # Try UTF-8 first, then Thai Windows (cp874)
        try:
            df = pd.read_csv(INPUT_FILE, encoding='utf-8')
        except:
            df = pd.read_csv(INPUT_FILE, encoding='cp874')
            
        print(f"📄 Processing {len(df)} products from CSV...")
        
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
            try:
                name = str(row.iloc[1]) # 'รายการ' is 2nd col
                sku = str(row.iloc[9]) if not pd.isna(row.iloc[9]) else "" # Barcode is last col
                
                prediction = self.classify(name)
                
                results.append({
                    "original_name": name,
                    "barcode": sku,
                    "suggested_category": prediction["category_name"],
                    "confidence": f"{prediction['confidence']:.2f}",
                    "method": prediction["method"],
                    "lesson_sim": f"{prediction['lesson_sim']:.2f}"
                })
            except Exception as e:
                print(f"❌ Error: {e}")
                
        # Save results
        out_df = pd.DataFrame(results)
        out_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"✅ Finished! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    processor = NewProductProcessor()
    processor.run()
