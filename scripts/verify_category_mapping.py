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
    
    # 2. Get unique categories from file (Col 8 and 9)
    # Using column names if they exist, otherwise indices
    main_col = df.columns[8]
    sub_col = df.columns[9]
    file_categories = df[[main_col, sub_col]].drop_duplicates()
    
    print("📊 Categories found in Reference File:")
    file_cat_list = []
    for _, row in file_categories.iterrows():
        main = str(row[main_col]).strip()
        sub = str(row[sub_col]).strip()
        print(f"- {main} > {sub}")
        file_cat_list.append(sub)

    # 3. Get taxonomy from DB
    supabase = create_client(url, key)
    res = supabase.table('taxonomy_nodes').select('id, name_th').execute()
    db_nodes = {node['name_th']: node['id'] for node in res.data}
    
    print(f"\n✅ Found {len(db_nodes)} nodes in Database.")
    
    # 4. Check mapping
    missing = []
    for sub in file_cat_list:
        if sub not in db_nodes:
            missing.append(sub)
            
    if missing:
        print(f"\n⚠️ Missing Categories in DB ({len(missing)}):")
        for m in missing:
            print(f"- {m}")
    else:
        print("\n✨ All file categories match Database nodes!")
        
except Exception as e:
    print(f"Error: {e}")
