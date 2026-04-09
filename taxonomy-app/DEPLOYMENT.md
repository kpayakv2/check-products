# 🚀 Deployment Guide - Thai Product Taxonomy Manager

## Prerequisites

- Node.js 18+ และ npm
- Supabase Account
- FastAPI Backend (api_server.py)
- Git

## 1. Supabase Setup

### สร้าง Supabase Project
1. ไปที่ [supabase.com](https://supabase.com)
2. สร้าง project ใหม่
3. รอให้ database พร้อมใช้งาน

### ติดตั้ง pgvector Extension
```sql
-- ใน SQL Editor ของ Supabase
CREATE EXTENSION IF NOT EXISTS vector;
```

### รัน Database Schema
```bash
# คัดลอกเนื้อหาจาก supabase/schema.sql
# แล้ววางใน SQL Editor ของ Supabase และรัน
```

## 2. Environment Variables

สร้างไฟล์ `.env.local`:

```env
# Supabase
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key

# Local AI Backend (FastAPI)
FASTAPI_URL=http://localhost:8000

# App Settings
NEXT_PUBLIC_APP_URL=http://localhost:3000
```

## 3. Deploy Edge Functions

### ติดตั้ง Supabase CLI
```bash
npm install -g supabase
```

### Login และ Link Project
```bash
supabase login
supabase link --project-ref your_project_ref
```

### Deploy Functions
```bash
# Deploy ทุก functions
supabase functions deploy

# หรือ deploy แต่ละตัว
supabase functions deploy hybrid-classification-local
supabase functions deploy generate-embeddings-local
supabase functions deploy hybrid-search
```

### ตั้งค่า Environment Variables สำหรับ Edge Functions
```bash
supabase secrets set FASTAPI_URL=http://host.docker.internal:8000
```

## 4. Local Development

### ติดตั้ง Dependencies
```bash
npm install
```

### รัน Development Server
```bash
npm run dev
```

### ทดสอบการเชื่อมต่อ
```bash
npm run test-connection
```

## 5. Production Deployment

### Vercel Deployment

1. **Push to GitHub**
```bash
git add .
git commit -m "Initial deployment"
git push origin main
```

2. **Deploy to Vercel**
- ไปที่ [vercel.com](https://vercel.com)
- Import GitHub repository
- ตั้งค่า Environment Variables
- Deploy

3. **Environment Variables ใน Vercel**
```
NEXT_PUBLIC_SUPABASE_URL
NEXT_PUBLIC_SUPABASE_ANON_KEY
SUPABASE_SERVICE_ROLE_KEY
OPENAI_API_KEY
HUGGINGFACE_API_KEY
NEXT_PUBLIC_APP_URL
```

### Netlify Deployment

1. **Build Settings**
```
Build command: npm run build
Publish directory: .next
```

2. **Environment Variables**
- ตั้งค่าเหมือน Vercel

## 6. Database Initialization

### สร้างข้อมูลเริ่มต้น

```sql
-- สร้าง taxonomy nodes ตัวอย่าง
INSERT INTO taxonomy_nodes (name_th, name_en, level, sort_order, keywords, is_active) VALUES
('อิเล็กทรอนิกส์', 'Electronics', 0, 1, ARRAY['อิเล็กทรอนิกส์', 'electronics', 'gadget'], true),
('เสื้อผ้า', 'Clothing', 0, 2, ARRAY['เสื้อผ้า', 'clothing', 'fashion'], true),
('อาหาร', 'Food', 0, 3, ARRAY['อาหาร', 'food', 'eat'], true);

-- สร้าง keyword rules ตัวอย่าง
INSERT INTO keyword_rules (name, description, keywords, category_id, match_type, priority, confidence_score, is_active) VALUES
('iPhone Detection', 'Detect iPhone products', ARRAY['iphone', 'ไอโฟน'], 
 (SELECT id FROM taxonomy_nodes WHERE name_th = 'อิเล็กทรอนิกส์'), 
 'contains', 10, 0.9, true);

-- สร้าง system settings เริ่มต้น
INSERT INTO system_settings (id) VALUES (gen_random_uuid());
```

## 7. Performance Optimization

### Database Indexes
```sql
-- สร้าง indexes สำหรับ performance
CREATE INDEX IF NOT EXISTS idx_products_embedding ON products USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category_id);
CREATE INDEX IF NOT EXISTS idx_products_status ON products(status);
CREATE INDEX IF NOT EXISTS idx_taxonomy_keywords ON taxonomy_nodes USING GIN(keywords);
```

### Caching Strategy
- ใช้ Vercel Edge Caching
- Cache taxonomy tree ใน localStorage
- Cache search results (5 นาที)

## 8. Monitoring & Analytics

### Supabase Analytics
- ดู API usage ใน Supabase Dashboard
- Monitor database performance
- ตรวจสอบ Edge Function logs

### Error Tracking
```typescript
// เพิ่มใน utils/monitoring.ts
export const trackError = (error: Error, context: string) => {
  console.error(`[${context}]`, error)
  // ส่งไป error tracking service
}
```

## 9. Security Checklist

### Row Level Security (RLS)
```sql
-- เปิดใช้งาน RLS
ALTER TABLE products ENABLE ROW LEVEL SECURITY;
ALTER TABLE taxonomy_nodes ENABLE ROW LEVEL SECURITY;

-- สร้าง policies
CREATE POLICY "Users can read products" ON products FOR SELECT USING (true);
CREATE POLICY "Authenticated users can insert products" ON products FOR INSERT WITH CHECK (auth.role() = 'authenticated');
```

### API Rate Limiting
```typescript
// ใน Edge Functions
const rateLimiter = new Map()
const RATE_LIMIT = 100 // requests per minute

const checkRateLimit = (userId: string) => {
  const now = Date.now()
  const userRequests = rateLimiter.get(userId) || []
  const recentRequests = userRequests.filter(time => now - time < 60000)
  
  if (recentRequests.length >= RATE_LIMIT) {
    throw new Error('Rate limit exceeded')
  }
  
  recentRequests.push(now)
  rateLimiter.set(userId, recentRequests)
}
```

## 10. Backup & Recovery

### Database Backup
```bash
# ใช้ Supabase CLI
supabase db dump --file backup.sql

# Restore
supabase db reset --file backup.sql
```

### Code Backup
- ใช้ Git สำหรับ version control
- สำรอง environment variables
- สำรอง Supabase project settings

## 11. Troubleshooting

### Common Issues

**1. Edge Functions ไม่ทำงาน**
```bash
# ตรวจสอบ logs
supabase functions logs hybrid-search

# ตรวจสอบ secrets
supabase secrets list
```

**2. Database Connection Error**
- ตรวจสอบ environment variables
- ตรวจสอบ Supabase project status
- ตรวจสอบ network connectivity

**3. Embedding Generation ล้มเหลว**
- ตรวจสอบ OpenAI API key
- ตรวจสอบ rate limits
- ลองใช้ Hugging Face แทน

**4. Search ไม่แสดงผล**
- ตรวจสอบ pgvector extension
- ตรวจสอบ indexes
- ตรวจสอบ RLS policies

## 12. Scaling Considerations

### Database Scaling
- ใช้ Supabase Pro สำหรับ production
- เพิ่ม connection pooling
- Optimize queries

### API Scaling
- ใช้ CDN สำหรับ static assets
- Implement caching strategies
- Monitor API usage

### Cost Optimization
- ใช้ Supabase free tier สำหรับ development
- Monitor OpenAI API usage
- Optimize embedding generation

---

## 🎯 Quick Start Commands

```bash
# 1. Clone และ setup
git clone <repository>
cd taxonomy-app
npm install

# 2. Setup environment
cp .env.local.example .env.local
# แก้ไข .env.local

# 3. Deploy Edge Functions
supabase login
supabase link --project-ref <your-ref>
supabase functions deploy

# 4. Run locally
npm run dev
```

## 📞 Support

หากมีปัญหาในการ deploy:
1. ตรวจสอบ logs ใน Supabase Dashboard
2. ตรวจสอบ Vercel deployment logs
3. ตรวจสอบ browser console สำหรับ client-side errors

Happy Deploying! 🚀
