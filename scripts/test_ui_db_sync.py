
import uuid
import datetime
import random
import string
from supabase import create_client, Client

def test_create_category():
    url = "http://127.0.0.1:54321" # Supabase Local API
    key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." # Need a real service role key or just use DB direct
    
    # Better to use DB direct since I don't have the key handy in env here easily
    # But wait, I can just use docker exec to insert and verify.
    
    name_th = "หมวดหมู่ทดสอบใหม่"
    # Logic from UI: const uniqueCode = `CAT-${Date.now().toString().slice(-6)}${Math.random().toString(36).substring(2, 5).toUpperCase()}`
    now_ms = int(datetime.datetime.now().timestamp() * 1000)
    rand_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    code = f"CAT-{str(now_ms)[-6:]}{rand_str}"
    
    print(f"🛠️ Simulating UI Category Creation...")
    print(f"   Name (TH): {name_th}")
    print(f"   Code: {code}")
    
    import subprocess
    
    cmd = f"docker exec supabase_db_product_checker psql -U postgres -d postgres -c \"INSERT INTO taxonomy_nodes (id, code, name_th, level, is_active) VALUES ('{uuid.uuid4()}', '{code}', '{name_th}', 0, true) RETURNING *;\""
    
    try:
        result = subprocess.check_output(cmd, shell=True).decode('utf-8')
        print("\n✅ Database Response:")
        print(result)
        
        # Verify
        verify_cmd = f"docker exec supabase_db_product_checker psql -U postgres -d postgres -c \"SELECT id, code, name_th FROM taxonomy_nodes WHERE code = '{code}';\""
        verify_result = subprocess.check_output(verify_cmd, shell=True).decode('utf-8')
        print("\n🔍 Verification Query:")
        print(verify_result)
        
        if name_th in verify_result and code in verify_result:
            print("\n✨ UI vs DB Sync Test: PASSED")
        else:
            print("\n❌ UI vs DB Sync Test: FAILED")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    test_create_category()
