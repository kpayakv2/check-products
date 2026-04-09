import os
import sys
from supabase import create_client

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

url = 'http://localhost:54331'
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

sql_file = 'taxonomy-app/new_keyword_rules_v4.sql'

try:
    with open(sql_file, 'r', encoding='utf-8') as f:
        sql = f.read()
    
    s = create_client(url, key)
    # Using rpc to execute multiple inserts at once
    res = s.rpc('exec_sql', {'query_text': sql}).execute()
    print(res.data)
except Exception as e:
    print(f"Error: {e}")
