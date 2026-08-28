import asyncio
import httpx
import sys
from supabase import create_client, Client
import os
from dotenv import load_dotenv

# Fix Windows console encoding (คอนโซลไทยเป็น cp874 พิมพ์ emoji ไม่ได้)
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# โหลด Environment Variables
load_dotenv('taxonomy-app/.env.local')
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") # ต้องใช้ Service Role เพื่อ Update
fastapi_url = "http://127.0.0.1:8000/api/embed"

async def generate_category_embeddings():
    supabase: Client = create_client(url, key)
    
    # 1. ดึงหมวดหมู่ที่ไม่มี Embedding
    print("🔍 Fetching categories without embeddings...")
    response = supabase.table("taxonomy_nodes").select("id, name_th, keywords").is_("embedding", "null").execute()
    categories = response.data
    
    if not categories:
        print("✅ All categories already have embeddings!")
        return

    print(f"🔄 Found {len(categories)} categories to process.")

    async with httpx.AsyncClient() as client:
        for cat in categories:
            name = cat['name_th']
            # ใช้ชื่อ + keywords ให้ตรงกับสูตรของ embedding เดิมใน DB
            # (เทียบแล้ว cos ~0.95-0.97 เทียบกับ name_th อย่างเดียวที่ได้แค่ 0.34-0.87)
            keywords = cat.get('keywords') or []
            text = ' '.join([name] + keywords)
            print(f"🧠 Generating embedding for: {name}")

            # 2. เรียก FastAPI เพื่อขอ Embedding
            try:
                res = await client.post(fastapi_url, json={"text": text}, timeout=30.0)
                if res.status_code == 200:
                    embedding = res.json()['embedding']
                    
                    # 3. อัปเดตกลับลง Supabase
                    supabase.table("taxonomy_nodes").update({"embedding": embedding}).eq("id", cat['id']).execute()
                    print(f"✅ Updated: {name}")
                else:
                    print(f"❌ Failed to get embedding for {name}: {res.text}")
            except Exception as e:
                print(f"⚠️ Error processing {name}: {e}")

if __name__ == "__main__":
    asyncio.run(generate_category_embeddings())
