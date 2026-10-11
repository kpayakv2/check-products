# สองเสาหลักสำหรับแหล่งความจริงของ Type (Postgres สำหรับ DB, Pydantic สำหรับ AI)

Frontend มีการประกาศ type ซ้ำซ้อนกัน 11 รายการใน 25 ไฟล์ทั่วทั้งโปรเจกต์ (`TaxonomyNode`, `Product`, `KeywordRule`, `ImportRecord`, ฯลฯ) ทำให้เกิด schema drift เรื้อรังระหว่างฐานข้อมูลกับหน้าบ้าน และระหว่าง FastAPI กับ Next.js เราจึงกำหนดให้ระบบมี**แหล่งความจริงของ Type สองเสาหลักแบ่งตามขอบเขตความรับผิดชอบ (Two-Pillar Single Source of Truth)** โดยไม่เขียน interface ซ้ำเองในระดับ component:

1. **เสา Database Schema (PostgreSQL):** โครงสร้างตารางใน PostgreSQL (`127.0.0.1:54325`) เป็นแหล่งความจริงหนึ่งเดียว ฝั่ง TypeScript จะสร้าง `taxonomy-app/types/database.types.ts` อัตโนมัติด้วยคำสั่ง `supabase gen types typescript` และส่งต่อ generic `<Database>` ให้กับ Supabase client (`supabase` และ `supabaseAdmin`)
2. **เสา AI Inference Services (FastAPI / Pydantic):** Pydantic models ใน Python (`src/api/models.py` และ routers) เป็นแหล่งความจริงหนึ่งเดียวของ AI APIs ฝั่ง TypeScript จะสร้าง `taxonomy-app/types/fastapi.types.ts` อัตโนมัติด้วยคำสั่ง `openapi-typescript` จาก `http://127.0.0.1:8000/openapi.json`
3. **การบริโภคใน Frontend (Next.js):** รวบรวม Application / View Model types ไว้ที่โฟลเดอร์ `taxonomy-app/types/` (เช่น `types/index.ts`, `types/import.ts`) โดย import จาก `database.types.ts` และ `fastapi.types.ts` ห้ามเขียน interface ตารางขึ้นมาเองในไฟล์ component
4. **กลไกการสร้าง:** เพิ่มคำสั่ง on-demand `npm run gen:types` ใน `package.json` ให้นักพัฒนารันเมื่อมีการแก้ DB schema หรือ Pydantic models แล้ว commit ไฟล์ generated types เข้า git โดยไม่ผูกกับ `prebuild` ใน Next.js เพื่อป้องกันไม่ให้คำสั่ง build ล้มเหลวเมื่อ DB หรือ FastAPI ไม่ได้เปิดอยู่

## Considered Options

- **ปล่อยแบบเดิม (ประกาศซ้ำในแต่ละไฟล์/คอมโพเนนต์):** ปฏิเสธ เพราะจากการตรวจพบ F-027 พบ 11 types กระจายใน 25 ไฟล์ ทำให้ฟิลด์ optional และ nullability คลาดเคลื่อนกันจริง จนเกิดบั๊กเงียบหลายครั้ง
- **Pydantic เป็นต้นฉบับอย่างเดียว:** ปฏิเสธ เพราะ FastAPI มีเพียง 6 endpoint ที่ใช้งานจริงในการทำ AI classification/embedding และไม่ได้เป็นเจ้าของตารางส่วนใหญ่ในฐานข้อมูล (เช่น `products`, `taxonomy_nodes`, `imports`)
- **TS / Zod เป็นต้นฉบับอย่างเดียว:** ปฏิเสธ เพราะแม้จะรวมศูนย์ฝั่ง frontend ได้ แต่ไม่สามารถป้องกัน drift ที่รอยต่อระหว่าง Next.js กับ Python AI pipeline ได้

## Consequences

- ลดโค้ด duplicate types ใน frontend จาก 25 ไฟล์ เหลือไฟล์นิยามกลางเพียงที่เดียว
- การแก้ไข schema ฝั่ง DB หรือ AI endpoint ในอนาคตจะสะท้อนเป็น TypeScript compile error ทันทีใน `tsc --noEmit` หากสัญญาเปลี่ยน
- ไม่เพิ่ม runtime overhead หรือ build-time dependency ที่เปราะบาง เพราะใช้ไฟล์ generated types ที่ commit ลงใน repository
