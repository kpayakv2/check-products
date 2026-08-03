import sys
import traceback

try:
    from src.api.dependencies import get_supabase_client
    client = get_supabase_client()
    res = client.table('similarity_matches').select('*').limit(5).execute()
    print("Database Connection Successful!")
    print(res)
except Exception as e:
    traceback.print_exc()
