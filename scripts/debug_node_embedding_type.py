import os
import json
import numpy as np
from supabase import create_client

url = 'http://localhost:54331'
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

try:
    s = create_client(url, key)
    res = s.table('taxonomy_nodes').select('name_th, embedding').limit(1).execute()
    
    if res.data:
        item = res.data[0]
        emb = item['embedding']
        print(f"Node: {item['name_th']}")
        print(f"Embedding Type: {type(emb)}")
            
except Exception as e:
    print(f"Error: {e}")
