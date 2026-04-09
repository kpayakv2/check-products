import os
import json
from supabase import create_client
from dotenv import load_dotenv

# Load from taxonomy-app/.env.local if needed
# but we can set them directly for this check
url = 'http://localhost:54331'
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

try:
    s = create_client(url, key)
    res = s.table('keyword_rules').select('id, name, keywords, category_id').execute()
    print(json.dumps(res.data, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"Error: {e}")
