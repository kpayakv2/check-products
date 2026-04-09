import os
import json
import sys
from supabase import create_client

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

url = os.getenv("SUPABASE_URL", 'http://localhost:54331')
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

try:
    s = create_client(url, key)
    res = s.table('taxonomy_nodes').select('id, name_th, keywords').execute()
    print(json.dumps(res.data, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"Error: {e}")
