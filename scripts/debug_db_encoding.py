import os
import json
from supabase import create_client

url = 'http://localhost:54331'
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

try:
    s = create_client(url, key)
    res = s.table('taxonomy_nodes').select('id, name_th').execute()
    with open('debug_db_names.json', 'w', encoding='utf-8') as f:
        json.dump(res.data, f, indent=2, ensure_ascii=False)
    print("✅ Saved DB names to debug_db_names.json")
except Exception as e:
    print(f"Error: {e}")
