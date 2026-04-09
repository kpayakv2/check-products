import os
import sys
import pandas as pd
from supabase import create_client
import uuid

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

file_path = 'input/รายการสินค้าพร้อมหมวดหมู่_AI.txt'
url = 'http://localhost:54331'
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

try:
    # 1. Read the Reference File
    df = pd.read_csv(file_path, sep='\t', encoding='utf-16')
    
    # Extract unique (Main, Sub) pairs
    # Col 8: Main, Col 9: Sub
    main_col = df.columns[8]
    sub_col = df.columns[9]
    pairs = df[[main_col, sub_col]].drop_duplicates().values.tolist()
    
    supabase = create_client(url, key)
    
    # 2. Get existing categories
    res = supabase.table('taxonomy_nodes').select('id, name_th, level').execute()
    db_nodes = {node['name_th'].strip(): node['id'] for node in res.data}
    
    print(f"📊 Current DB has {len(db_nodes)} categories.")
    print(f"📊 Reference file has {len(pairs)} category pairs.")
    
    # 3. Create missing categories
    new_nodes_count = 0
    
    for main, sub in pairs:
        main = str(main).strip()
        sub = str(sub).strip()
        
        # Check/Create Main Category (Level 0)
        if main not in db_nodes:
            print(f"➕ Creating Main Category: {main}")
            new_main = supabase.table('taxonomy_nodes').insert({
                "name_th": main,
                "level": 0,
                "is_active": True
            }).execute().data[0]
            db_nodes[main] = new_main['id']
            new_nodes_count += 1
            
        parent_id = db_nodes[main]
        
        # Check/Create Sub Category (Level 1)
        if sub not in db_nodes:
            print(f"  ➕ Creating Sub Category: {sub} (Parent: {main})")
            new_sub = supabase.table('taxonomy_nodes').insert({
                "name_th": sub,
                "level": 1,
                "parent_id": parent_id,
                "is_active": True
            }).execute().data[0]
            db_nodes[sub] = new_sub['id']
            new_nodes_count += 1
            
    print(f"✅ Created {new_nodes_count} new category nodes.")
    
except Exception as e:
    print(f"Error: {e}")
