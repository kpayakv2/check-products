# 🏗️ API Architecture (Single Source of Truth)

**อัปเดตล่าสุด:** 2026-10-11
**หลักการสำคัญ:** 5-Layer Clean Architecture & Two-Pillar Types SSOT ([ADR 0004](../adr/0004-two-pillar-single-source-of-truth-types.md))

---

## 📊 ภาพรวมสถาปัตยกรรม (System Architecture)

```mermaid
graph TD
    subgraph Client
        Browser[Web Browser / LAN Clients]
    end

    subgraph "Next.js Frontend (Port 3000)"
        MW[Edge Middleware: Cookie Gate]
        Pages[App Router Pages]
        Rewrites[Next.js Rewrites Proxy]
        AdminRoutes[Next.js API Routes: Service Role Writes]
    end

    subgraph "FastAPI AI Engine (Port 8000)"
        FastAPI[FastAPI Backend]
        Embeddings[Sentence Transformer: 384-dim]
        Classifier[Hybrid Classification Engine]
        MLSystem[Continuous Learning System]
    end

    subgraph "Supabase Backend (Local Ports 54331 / 54325)"
        Gateway[API Gateway / Kong: 54331]
        Storage[Storage: uploads Bucket]
        Postgres[(PostgreSQL: 54325 + pgvector)]
    end

    Browser --> MW
    MW --> Pages
    MW --> Rewrites
    MW --> AdminRoutes

    Rewrites -->|/api/fastapi/*| FastAPI
    Rewrites -->|/api/supabase/*| Gateway

    AdminRoutes -->|supabaseAdmin (Bypass RLS)| Postgres
    Pages -->|Anon Client (Read-only)| Gateway
    Gateway --> Postgres
    Gateway --> Storage

    FastAPI --> Embeddings
    FastAPI --> Classifier
    FastAPI --> MLSystem
    FastAPI -->|Direct Connection| Postgres
    FastAPI -->|Download CSV| Storage
```

---

## 🏛️ สถาปัตยกรรม 5 ชั้นของระบบ (5-Layer Architecture)

### 1. Presentation & Proxy Layer (Next.js 14)
- **Role:** นำเสนอ UI จัดการ interaction ของผู้ใช้งาน และทำหน้าที่เป็น Reverse Proxy สำหรับ LAN Access ผ่าน Next.js rewrites:
  - `/api/fastapi/:path*` → `http://127.0.0.1:8000/:path*`
  - `/api/supabase/:path*` → `http://127.0.0.1:54331/:path*`
- ป้องกันปัญหา CORS และทำให้ Client บนเครือข่ายสำนักงาน (LAN) เข้าถึง API ได้ผ่าน Port 3000 เดียว

### 2. Next.js Mutation & Boundary Layer (Layer 2 API Routes)
- **Role:** ดูแลงานเขียนข้อมูลที่ต้องใช้สิทธิ์ผู้ดูแลระบบ (Admin Writes) และการตรวจสอบความถูกต้องของข้อมูล (Zod Validation)
- ใช้ `supabaseAdmin` (Service Role Key) สำหรับเขียน/แก้ไข/ลบข้อมูลเพื่อแก้ปัญหา RLS blocking และรับประกันความสม่ำเสมอ
- ได้รับการคุ้มครองด้วย `middleware.ts` ตรวจสอบ Token Session จากคุกกี้ที่ได้จาก `/unlock`

### 3. AI & ML Inference Layer (FastAPI 127.0.0.1:8000)
- **Role:** เครื่องมือคำนวณ Heavy-lifting ด้าน AI และ Machine Learning:
  - **Embedding Provider:** สร้าง Vector 384 มิติ (`paraphrase-multilingual-MiniLM-L12-v2`) สำหรับข้อความภาษาไทย
  - **Hybrid Classification Engine:** จำแนกหมวดหมู่สินค้าด้วยคะแนนผสม (Keyword Rules 60% + Cosine Vector 40%)
  - **Pairwise Scan & Deduplication:** ตรวจจับคู่สินค้าซ้ำซ้อนในคลังข้อมูลแบบ vectorized cosine dot-product
  - **Continuous Learning System:** เรียนรู้จากการตรวจสอบของมนุษย์ (Human Feedback) และเทรนโมเดล LightGBM/RandomForest แบบอัตโนมัติ

### 4. Database & Storage Layer (Supabase / PostgreSQL 127.0.0.1:54325)
- **Role:** แหล่งความจริงเดียวของข้อมูล (Single Source of Truth - SSOT)
- จัดเก็บ Taxonomy Nodes, Keywords, สินค้า, กฎเกณฑ์, และประวัติการรีวิว
- ค้นหาความคล้ายคลึงด้วย `pgvector` Cosine Distance (`<=>`)
- จัดเก็บไฟล์นำเข้า (CSV/Excel) ใน Storage Bucket `uploads`

### 5. Type System Infrastructure (ADR 0004)
- **Pillar 1:** `types/database.types.ts` สร้างอัตโนมัติจากตาราง PostgreSQL จริง
- **Pillar 2:** `types/fastapi.types.ts` สร้างอัตโนมัติจาก OpenAPI schema ของ FastAPI
- **Central SSOT:** `types/index.ts` รวบรวมและ export types ทั้งหมด ห้ามสร้าง local type interface ซ้ำซ้อน

---

## 🚫 การยกเลิก Supabase Edge Functions และ WebSocket

1. **Edge Functions (Layer 3):** ถูกถอดออกทั้งหมด (Purged) เนื่องจากเป็นเพียง proxy ซ้ำซ้อนที่ส่งต่อคำขอเข้า FastAPI และไม่มี edge runtime container รันใน local docker compose โดย Next.js เปลี่ยนไปคุยกับ FastAPI โดยตรงผ่าน rewrites
2. **WebSocket (Layer 5):** ถูกถอดออกทั้งหมด (`WS /ws`, `custom_websockets.py`) เนื่องจากระบบ UI ใช้ HTTP Polling / Task Status API แบบ idempotent ที่เสถียรกว่าบน Windows

---

## 🔐 สรุปนโยบายความปลอดภัย (Security Model)

- **Authentication:** โครงการใช้ Single Shared Office Unlock Token ที่สร้างจาก `sha256(INTERNAL_API_SECRET)` ผ่านหน้า `/unlock`
- **Route Guard:** `middleware.ts` สกัดกั้นทุก request ที่มีการกลายพันธุ์ (POST/PUT/DELETE) รวมถึงเส้นทางอ่าน service-role หากไม่มี valid session token
- **Database Access:** การอ่านข้อมูลใช้ Anon Key ผ่าน PostgREST, การเขียนข้อมูลทั้งหมดต้องทำผ่าน Service Role Route ใน Next.js
