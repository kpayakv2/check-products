import os
import json
import sys
from supabase import create_client

# Set encoding for Windows terminal
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

url = 'http://localhost:54331'
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

try:
    s = create_client(url, key)
    
    # Fetch taxonomy nodes
    nodes_res = s.table('taxonomy_nodes').select('id, name_th').execute()
    nodes_map = {node['id']: node['name_th'] for node in nodes_res.data}
    
    # Fetch keyword rules
    rules_res = s.table('keyword_rules').select('id, name, keywords, category_id').execute()
    
    results = []
    for rule in rules_res.data:
        rule['category_name'] = nodes_map.get(rule['category_id'], 'Unknown')
        results.append(rule)
        
    print(json.dumps(results, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"Error: {e}")
