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
from src.core.advanced_models import SentenceTransformerModel
from src.core.fresh_implementations import ThaiTextProcessor, ComponentFactory
from src.services.taxonomy_service import TaxonomyService
from src.core.scoring_logic import calculate_hybrid_score

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "http://127.0.0.1:54331")
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
        self.taxonomy = TaxonomyService(supabase)
        self.sim_calc = ComponentFactory.create_similarity_calculator("cosine")
        
    def load_metadata(self):
        self.taxonomy.load_all_metadata()

    def classify(self, name: str) -> dict:
        clean_name = self.processor.process(name)
        prod_emb = self.model.encode([clean_name])[0]
        
        # 1. Similarity with Lessons
        best_lesson_sim = 0.0
        best_lesson_cat = None
        
        for lesson in self.taxonomy.reference_lessons:
            lesson_emb = lesson['embedding_arr']
            sim = self.sim_calc.calculate(prod_emb, lesson_emb)
            if sim > best_lesson_sim:
                best_lesson_sim = sim
                best_lesson_cat = lesson['category_id']
        
        # 2. Keyword Match
        best_kw_cat = None
        best_kw_conf = 0.0
        for rule in self.taxonomy.keyword_rules:
            for kw in rule['keywords']:
                if self.processor.process(kw) in clean_name:
                    conf = (rule.get('confidence_score', 0.8))
                    if conf > best_kw_conf:
                        best_kw_conf = conf
                        best_kw_cat = rule['category_id']
        
        # 3. Hybrid Calculation
        best_cat_emb_sim = 0.0
        best_cat_emb_id = None
        for cat_id, cat_emb in self.taxonomy.category_embeddings.items():
            sim = self.sim_calc.calculate(prod_emb, cat_emb)
            if sim > best_cat_emb_sim:
                best_cat_emb_sim = sim
                best_cat_emb_id = cat_id
        
        # Final Decision
        if best_lesson_sim > 0.94:
            final_cat_id = best_lesson_cat
            final_conf = best_lesson_sim
            method = "lesson_match"
        else:
            if best_kw_cat:
                final_cat_id = best_kw_cat
                final_conf = calculate_hybrid_score(best_kw_conf, best_cat_emb_sim)
                method = "hybrid"
            else:
                final_cat_id = best_cat_emb_id
                final_conf = best_cat_emb_sim
                method = "embedding"
            
        return {
            "category_id": final_cat_id,
            "category_name": self.taxonomy.get_category_name(final_cat_id),
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
