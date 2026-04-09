# Setup Guide - Thai Product Taxonomy Manager

## 🚀 การติดตั้งและเชื่อมต่อ Supabase

### 1. ติดตั้ง Dependencies

```bash
cd taxonomy-app
npm install
```

### 2. ตั้งค่า Supabase

#### 2.1 สร้าง Supabase Project
1. ไปที่ [supabase.com](https://supabase.com)
2. สร้าง Project ใหม่
3. รอให้ Database พร้อมใช้งาน

#### 2.2 เปิดใช้งาน Extensions
รันคำสั่งใน SQL Editor ของ Supabase:

```sql
-- เปิดใช้งาน pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

#### 2.3 รัน Database Schema
คัดลอกเนื้อหาจากไฟล์ `supabase/schema.sql` และรันใน SQL Editor

### 3. ตั้งค่า Environment Variables

สร้างไฟล์ `.env.local`:

```bash
cp .env.local.example .env.local
```

แก้ไขไฟล์ `.env.local`:

```env
# Supabase Configuration
NEXT_PUBLIC_SUPABASE_URL=https://your-project-id.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# App Configuration
NEXT_PUBLIC_APP_NAME="Thai Product Taxonomy Manager"
NEXT_PUBLIC_APP_VERSION="1.0.0"
```

### 4. รันแอปพลิเคชัน

```bash
npm run dev
```

เปิดเบราว์เซอร์ไปที่: `http://localhost:3000`

## 📊 Database Schema Overview

### หารางหลัก:

#### `taxonomy_nodes`
- โครงสร้างหมวดหมู่แบบ hierarchical
- รองรับ keywords array สำหรับการค้นหา
- มี materialized path สำหรับ query ที่เร็ว

#### `synonyms` + `synonym_terms`
- จัดเก็บกลุ่มคำพ้องความหมาย
- รองรับหลายภาษา (ไทย/อังกฤษ)
- ระบบ confidence score

#### `products`
- ข้อมูลสินค้าพร้อม vector embedding (768 dimensions)
- รองรับ keywords array
- ระบบ status และ review workflow

#### `keyword_rules`
- กฎการจัดหมวดหมู่อัตโนมัติ
- รองรับ regex, fuzzy matching
- ระบบ priority

#### `similarity_matches`
- ผลการเปรียบเทียบความคล้าย
- รองรับหลาย algorithm (cosine, euclidean, jaccard)
- ระบบตรวจจับสินค้าซ้ำ

### Indexes ที่สำคัญ:

- **GIN Index** บน `taxonomy_nodes.keywords` และ `products.keywords`
- **IVFFlat Index** บน `products.embedding` สำหรับ vector search
- **Partial Index** สำหรับ active records

### Security:

- **Row Level Security (RLS)** เปิดใช้งานทุกตาราง
- **Roles**: `taxonomy_reader`, `taxonomy_editor`, `taxonomy_admin`
- **Audit Logging** อัตโนมัติสำหรับการเปลี่ยนแปลงข้อมูล

## 🔧 การใช้งาน API

### Taxonomy Management
```typescript
// ดึงข้อมูล taxonomy tree
const tree = await DatabaseService.getTaxonomyTree()

// สร้างหมวดหมู่ใหม่
const newNode = await DatabaseService.createTaxonomyNode({
  name_th: 'หมวดหมู่ใหม่',
  name_en: 'New Category',
  parent_id: 'parent-uuid',
  keywords: ['keyword1', 'keyword2']
})
```

### Synonym Management
```typescript
// ดึงข้อมูล synonyms
const synonyms = await DatabaseService.getSynonyms()

// สร้าง synonym group
const synonymGroup = await DatabaseService.createSynonym({
  name: 'โทรศัพท์มือถือ',
  description: 'กลุ่มคำเกี่ยวกับมือถือ'
})
```

### Product Review
```typescript
// ดึงสินค้ารอตรวจสอบ
const products = await DatabaseService.getProducts('pending')

// อนุมัติสินค้า
await DatabaseService.updateProductStatus(productId, 'approved', reviewerId)
```

## 🎨 UI Components

### Layout Components
- `Sidebar` - Navigation แบบ responsive
- `Header` - Top bar พร้อม notifications
- `TaxonomyTree` - Interactive tree view

### Premium Design System
- **Colors**: Blue gradient theme
- **Fonts**: Noto Sans Thai + Inter
- **Animations**: Framer Motion
- **Icons**: Lucide React

## 🚀 Production Deployment

### Vercel (แนะนำ)
```bash
npm run build
vercel --prod
```

### Environment Variables สำหรับ Production
- `NEXT_PUBLIC_SUPABASE_URL`
- `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`

## 🔍 Troubleshooting

### ปัญหาที่พบบ่อย:

1. **Vector Extension Error**
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

2. **RLS Policy Error**
   - ตรวจสอบว่า user มี role ที่ถูกต้อง
   - ตรวจสอบ RLS policies

3. **TypeScript Errors**
   ```bash
   npm install @types/node --save-dev
   ```

4. **Font Loading Issues**
   - ตรวจสอบ Google Fonts connection
   - ใช้ font fallback

## 📞 Support

หากมีปัญหา:
1. ตรวจสอบ Console logs
2. ตรวจสอบ Supabase logs
3. ตรวจสอบ Network requests
4. ดู Documentation ใน `/docs` folder
