import os
import json
import sys
from supabase import create_client

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

url = 'http://localhost:54331'
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

try:
    s = create_client(url, key)
    
    # Fetch taxonomy nodes
    nodes_res = s.table('taxonomy_nodes').select('id, name_th').execute()
    nodes_ids = {node['id'] for node in nodes_res.data}
    
    # Fetch keyword rules
    rules_res = s.table('keyword_rules').select('category_id').execute()
    rules_cat_ids = {rule['category_id'] for rule in rules_res.data}
    
    missing_ids = nodes_ids - rules_cat_ids
    
    missing_nodes = [node for node in nodes_res.data if node['id'] in missing_ids]
    
    print(f"Total nodes: {len(nodes_ids)}")
    print(f"Nodes with rules: {len(rules_cat_ids)}")
    print(f"Nodes missing rules: {len(missing_ids)}")
    print("\nMissing Nodes:")
    for node in missing_nodes:
        print(f"- {node['name_th']} ({node['id']})")
        
except Exception as e:
    print(f"Error: {e}")
