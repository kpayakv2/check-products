import os
import sys
import pandas as pd
from supabase import create_client

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

file_path = 'input/รายการสินค้าพร้อมหมวดหมู่_AI.txt'
url = os.getenv("SUPABASE_URL", 'http://localhost:54331')
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

try:
    # 1. Read the Reference File
    df = pd.read_csv(file_path, sep='\t', encoding='utf-16')
    
    # Get unique categories from file (Col 9 is Sub-category)
    sub_col_idx = 9
    sub_col_name = df.columns[sub_col_idx]
    file_sub_cats = df[sub_col_name].dropna().unique()
    
    # 2. Get taxonomy from DB
    supabase = create_client(url, key)
    res = supabase.table('taxonomy_nodes').select('id, name_th').execute()
    db_nodes = {node['name_th'].strip(): node['id'] for node in res.data}
    
    print(f"📊 Analyzing {len(file_sub_cats)} unique sub-categories from file...")
    
    matches = 0
    mismatches = []
    
    for f_cat in file_sub_cats:
        f_cat_clean = str(f_cat).strip()
        # Direct match
        if f_cat_clean in db_nodes:
            matches += 1
        else:
            # Fuzzy match (handle slash variations)
            f_cat_fuzzy = f_cat_clean.replace(' / ', '/').replace(' /', '/').replace('/ ', '/')
            found = False
            for db_cat in db_nodes:
                db_cat_fuzzy = db_cat.replace(' / ', '/').replace(' /', '/').replace('/ ', '/')
                if f_cat_fuzzy == db_cat_fuzzy:
                    matches += 1
                    found = True
                    break
            
            if not found:
                mismatches.append(f_cat_clean)
                
    print(f"✅ Exact/Fuzzy Matches: {matches}")
    print(f"❌ Mismatches: {len(mismatches)}")
    
    if mismatches:
        print("\n🔍 Mismatch Details (File -> DB suggestion?):")
        for m in mismatches[:15]:
            print(f"- '{m}'")
            
except Exception as e:
    print(f"Error: {e}")
