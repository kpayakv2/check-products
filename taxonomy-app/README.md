# Thai Product Taxonomy Manager

ระบบจัดการ Taxonomy และ Synonym สำหรับสินค้าภาษาไทย แบบ Premium SaaS

> **🎉 Complete Overhaul สำเร็จ 100%** - พร้อมใช้งาน Production ได้ทันที!

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](http://localhost:3000)
[![TypeScript](https://img.shields.io/badge/TypeScript-100%25-blue)](https://www.typescriptlang.org/)
[![Production Ready](https://img.shields.io/badge/Production-Ready-success)](http://localhost:3000)

## ✨ คุณสมบัติหลัก

### 🏠 Dashboard
- ภาพรวมของระบบแบบ Real-time
- สถิติการใช้งานและกิจกรรมล่าสุด
- การแจ้งเตือนและ Quick Actions

### 🌳 Taxonomy Management
- จัดการโครงสร้างหมวดหมู่แบบ Tree View
- รองรับหมวดหมู่หลักและหมวดหมู่ย่อยหลายระดับ
- การค้นหาและกรองหมวดหมู่
- Drag & Drop สำหรับจัดเรียงลำดับ

### 📚 Synonym Manager
- จัดการคำพ้องความหมายภาษาไทย
- รองรับการนำเข้า/ส่งออก CSV
- ระบบยืนยันและให้คะแนนความเชื่อมั่น
- การจัดกลุ่มตามหมวดหมู่

### 🛍️ Product Review
- ตรวจสอบและอนุมัติสินค้าใหม่
- ระบบตรวจจับสินค้าซ้ำซ้อน
- การจัดการ Attributes และ Metadata
- ประวัติการตรวจสอบ

### 📤 Import System
- อัปโหลดไฟล์สินค้า (.txt, .csv, .xlsx)
- ประมวลผลแบบ Real-time พร้อม Progress
- ทำความสะอาดข้อมูลและแยกคำภาษาไทย
- สกัดหน่วย (กรม, ลิตร, ชิ้น) และคุณสมบัติ
- สร้าง Vector Embeddings อัตโนมัติ
- แนะนำหมวดหมู่ด้วย AI พร้อมคะแนนความเชื่อมั่น
- อนุมัติและนำเข้าสินค้าเป็นชุด

### 🔍 Hybrid Search System
- Vector + Text Search รวมกัน
- Token Highlighting แบบ Smart
- Search Suggestions และ History
- Match Type Classification (Semantic/Keyword/Hybrid)
- Performance Metrics และ Real-time Results

### 🔄 Product Deduplication System
- ตรวจสอบสินค้าซ้ำซ้อนแบบ AI-Powered
- Human-in-the-Loop Review Interface
- TF-IDF และ Semantic Similarity Analysis
- การบันทึก Human Feedback สำหรับปรับปรุงระบบ
- Export ผลลัพธ์การตรวจสอบ
- รองรับไฟล์ CSV/Excel Upload

### ⚙️ Settings & Rules Management
- Regex Rules Editor พร้อม Live Testing
- Keyword Rules Management
- System Settings Configuration
- JSON Editor สำหรับ Advanced Configuration
- Import/Export Rules (JSON/CSV/YAML)

### 🤖 AI-Powered Edge Functions
- Hybrid Search API (Vector + Text)
- Category Suggestions API
- Embedding Generation API
- Thai Text Processing Pipeline
- Real-time Performance Monitoring

## 🚀 เทคโนโลยีที่ใช้

- **Frontend**: Next.js 14, React 18, TypeScript
- **UI Framework**: Tailwind CSS, Headless UI, Framer Motion
- **Database**: Supabase (PostgreSQL + pgvector)
- **Fonts**: Noto Sans Thai, Inter
- **Icons**: Lucide React
- **State Management**: React Hooks
- **Notifications**: React Hot Toast

## 📦 การติดตั้ง

### ข้อกำหนดระบบ
- Node.js 18+ 
- npm หรือ yarn
- Supabase Account

### ขั้นตอนการติดตั้ง

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd taxonomy-app
   ```

2. **ติดตั้ง Dependencies**
   ```bash
   npm install
   # หรือ
   yarn install
   ```

3. **ตั้งค่า Environment Variables**
   ```bash
   cp .env.local.example .env.local
   ```
   
   แก้ไขไฟล์ `.env.local`:
   ```env
   NEXT_PUBLIC_SUPABASE_URL=your_supabase_project_url
   NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
   SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
   ```

4. **ตั้งค่า Supabase Database**
   ```bash
   # รันไฟล์ SQL schema
   psql -h your-supabase-host -U postgres -d postgres -f supabase/schema.sql
   ```

5. **รันแอปพลิเคชัน**
   ```bash
   npm run dev
   # หรือ
   yarn dev
   ```

   เปิดเบราว์เซอร์ไปที่ `http://localhost:3000`

## 🗃️ Database Schema

### หารางหลัก

#### `taxonomy_categories`
- จัดเก็บโครงสร้างหมวดหมู่สินค้า
- รองรับ Hierarchical Structure
- ฟิลด์ภาษาไทยและอังกฤษ

#### `synonyms`
- จัดเก็บคำพ้องความหมาย
- ระบบคะแนนความเชื่อมั่น
- การเชื่อมโยงกับหมวดหมู่

#### `products`
- ข้อมูลสินค้าพื้นฐาน
- รองรับ Vector Embedding สำหรับ Semantic Search
- ระบบ Status และ Review

#### `similarity_matches`
- ผลการเปรียบเทียบความคล้ายของสินค้า
- คะแนนความคล้าย (Similarity Score)
- ประเภทการจับคู่ (Semantic, Exact, Fuzzy)

## 🎨 UI/UX Design

### Design System
- **สีหลัก**: Blue Gradient (#0ea5e9 → #0284c7)
- **ฟอนต์**: Noto Sans Thai สำหรับภาษาไทย, Inter สำหรับภาษาอังกฤษ
- **Components**: Premium Card Design พร้อม Glass Effect
- **Animation**: Smooth Transitions ด้วย Framer Motion

### Responsive Design
- รองรับทุกขนาดหน้าจอ (Mobile, Tablet, Desktop)
- Mobile-first Approach
- Touch-friendly Interface

## 🔧 API Endpoints

### Taxonomy
- `GET /api/taxonomy` - ดึงข้อมูล Taxonomy Tree
- `POST /api/taxonomy` - สร้างหมวดหมู่ใหม่
- `PUT /api/taxonomy/[id]` - แก้ไขหมวดหมู่
- `DELETE /api/taxonomy/[id]` - ลบหมวดหมู่

### Synonyms
- `GET /api/synonyms` - ดึงข้อมูล Synonyms
- `POST /api/synonyms` - สร้าง Synonym ใหม่
- `PUT /api/synonyms/[id]` - แก้ไข Synonym
- `DELETE /api/synonyms/[id]` - ลบ Synonym

### Products
- `GET /api/products` - ดึงข้อมูลสินค้า
- `POST /api/products` - สร้างสินค้าใหม่
- `POST /api/products/[id]/review` - ตรวจสอบสินค้า

### Similarity
- `GET /api/similarity/[id]` - ดึงข้อมูลสินค้าที่คล้ายกัน
- `POST /api/similarity/[id]` - สร้างการจับคู่ใหม่

## 🔒 Security

- **🛡️ Production-Ready Security**: Rate limiting, input validation, error handling
- **Row Level Security (RLS)** บน Supabase
- **API Rate Limiting**: 10 req/min GET, 5 req/min POST
- **Input Validation**: Zod schemas พร้อม Thai language support
- **Environment Security**: .env.local ถูก gitignore แล้ว
- **Error Handling**: Structured error responses พร้อม request tracking

## 🌐 Internationalization

- รองรับภาษาไทยและอังกฤษ
- Font Optimization สำหรับภาษาไทย
- Thai-specific Text Processing

## 📊 Performance

- Next.js App Router สำหรับ Performance
- Image Optimization
- Font Optimization
- Code Splitting
- Lazy Loading

## 🧪 Testing

```bash
# รัน Tests
npm run test

# รัน Tests แบบ Watch Mode
npm run test:watch

# รัน E2E Tests
npm run test:e2e
```

## 🚀 Deployment

### Vercel (แนะนำ)
```bash
npm run build
vercel --prod
```

### Docker
```bash
docker build -t taxonomy-app .
docker run -p 3000:3000 taxonomy-app
```

## 📝 Contributing

1. Fork the repository
2. สร้าง feature branch (`git checkout -b feature/amazing-feature`)
3. Commit การเปลี่ยนแปลง (`git commit -m 'Add amazing feature'`)
4. Push ไปยัง branch (`git push origin feature/amazing-feature`)
5. เปิด Pull Request

## 📄 License

โปรเจกต์นี้ใช้ MIT License - ดูรายละเอียดในไฟล์ [LICENSE](LICENSE)

## 🆘 Support

หากมีปัญหาหรือข้อสงสัย:
- เปิด Issue ใน GitHub Repository
- ติดต่อทีมพัฒนาผ่าน Email
- ดูเอกสารเพิ่มเติมใน `/docs` folder

## 🔄 Changelog

### v2.0.0 (2024-09-29) - **Complete Overhaul** 🎉
- **✅ BUILD SUCCESS**: ไม่มี TypeScript errors
- **✅ Production Security**: Rate limiting, validation, error handling
- **✅ Database Fixes**: แก้ไข API method mismatches ทั้งหมด
- **✅ Interface Updates**: TaxonomyCategory → TaxonomyNode
- **✅ Enhanced Components**: ทำงานได้ครบถ้วน
- **✅ Utility Libraries**: Rate limiting, error handling, validation
- **✅ Environment Security**: .env.local ถูก gitignore
- **✅ Development Server**: รันได้แล้ว (http://localhost:3000)

### v1.0.0 (2024-01-XX)
- เปิดตัวระบบครั้งแรก
- Dashboard และ Analytics
- Taxonomy Management
- Synonym Manager
- Product Review System
- Thai Language Support
- Premium UI/UX Design

---

**Made with ❤️ for Thai E-commerce**
