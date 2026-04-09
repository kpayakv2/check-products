import os
import json
import numpy as np
from supabase import create_client

url = 'http://localhost:54331'
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

try:
    s = create_client(url, key)
    res = s.table('products').select('name_th, embedding').limit(1).execute()
    
    if res.data:
        item = res.data[0]
        emb = item['embedding']
        print(f"Product: {item['name_th']}")
        print(f"Embedding Type: {type(emb)}")
        if isinstance(emb, str):
            print(f"Embedding is STRING: {emb[:50]}...")
        elif isinstance(emb, list):
            print(f"Embedding is LIST: {len(emb)} elements")
            print(f"First element type: {type(emb[0])}")
            
except Exception as e:
    print(f"Error: {e}")
