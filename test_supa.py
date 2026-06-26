import sys; import traceback; try:
  from src.api.dependencies import get_supabase_client; client=get_supabase_client(); print(client.table('similarity_matches').select('*').limit(5).execute())
except Exception as e:
  traceback.print_exc()
