# 🧠 Memory MCP: บทเรียนสถาปัตยกรรม API 5 ชั้น, สัญญาประเภท Two-Pillar SSOT และการชำระล้างระบบ
*บันทึกข้อมูล ณ วันที่: 2026-10-11*

## 1. Two-Pillar Single Source of Truth for Types (ADR 0004)
* **ปัญหาดั้งเดิม**: มีการประกาศ `interface` ซ้ำซ้อนกระจัดกระจาย 11 แห่งใน 25 ไฟล์ของ frontend (`RecheckTab`, `VerifyTab`, `AutoLearnTab`, `StorageImport`, `pos-export` ฯลฯ) ทำให้เมื่อ schema ฐานข้อมูลหรือ API payload เปลี่ยนแปลง โค้ดแต่ละจุดไม่ซิงค์กันและเกิด silent runtime type mismatch
* **บทเรียนการแก้ไข**:
  - เสาหลักที่ 1 (Database Schema): สร้าง `taxonomy-app/types/database.types.ts` จาก PostgreSQL จริง (พอร์ต 54325) ผ่านคำสั่ง `supabase gen types typescript`
  - เสาหลักที่ 2 (AI Service Contracts): สร้าง `taxonomy-app/types/fastapi.types.ts` จาก FastAPI OpenAPI schema จริง (พอร์ต 8000) ผ่านคำสั่ง `openapi-typescript`
  - สร้างไฟล์รวมศูนย์ [taxonomy-app/types/index.ts](file:///d:/product_checker/check-products/taxonomy-app/types/index.ts) ที่ re-export domain/view types ที่แท้จริง และห้ามคอมโพเนนต์ใด ๆ ใน frontend นิยาม type อิสระเองซ้ำ
  - เพิ่มคำสั่ง `npm run gen:types` ใน `taxonomy-app/package.json` สำหรับรัน generate อัตโนมัติเมื่อใดก็ตามที่มีการแก้ DB schema หรือ Pydantic models

---

## 2. Pydantic v2.13 (Python 3.14) ข้อควรระวังระหว่าง `Config` กับ `model_config`
* **ข้อผิดพลาด**: `pydantic.errors.PydanticUserError: "Config" and "model_config" cannot be used together`
* **สาเหตุ**: ใน Pydantic v2.13 ขึ้นไป หากประกาศทั้งคลาสภายใน `class Config:` และตัวแปร `model_config = ...` ภายในคลาสเดียวกัน Pydantic จะ throw `PydanticUserError` ทันทีที่โหลดโมดูล
* **บทเรียนการแก้ไข**:
  - ใช้ `model_config = ConfigDict(populate_by_name=True)` หรือ `model_config = {"populate_by_name": True}` เพียงอย่างเดียว ห้ามใส่ `class Config` ควบคู่กัน
  - เมื่อเปลี่ยนชื่อฟิลด์จาก `camelCase` เป็น `snake_case` (เช่น `processingTime` -> `processing_time`) ให้ใส่ `@property` และ `@name.setter` สำหรับชื่อเดิม เพื่อรักษาความเข้ากันได้แบบ 100% Backward Compatibility กับโค้ดและชุดทดสอบเดิม

---

## 3. Zod UUID Validation vs Postgres UUID
* **ข้อผิดพลาด**: Zod's `z.string().uuid()` ปฏิเสธค่า mock UUID ในการทดสอบ เช่น `11111111-1111-1111-1111-111111111111` แม้จะเป็นค่าที่ Postgres ยอมรับได้
* **สาเหตุ**: Zod ตรวจสอบตามมาตรฐาน RFC 4122 โดยเฉพาะ Version/Variant bits ทำให้ UUID ตัวเลขเรียงหรือ Mock UUIDs ไม่ผ่านการตรวจ
* **บทเรียนการแก้ไข**:
  - กำหนด Schema สำหรับตรวจ UUID ของฐานข้อมูลด้วย Regex แทน:
    ```typescript
    export const uuidSchema = z.string().regex(
      /^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/,
      'Invalid UUID format'
    )
    ```

---

## 4. Supabase Storage Bucket ในเครื่องพัฒนา (Local Instance)
* **ปัญหาดั้งเดิม**: หน้าเว็บ `StorageImport` และ `ImportHistory` เรียกใช้ bucket `uploads` แต่ในเครื่องไม่มี bucket ดังกล่าว ทำให้ PostgREST คืน 404/error เมื่อพยายามอัปโหลดหรือดาวน์โหลดไฟล์
* **บทเรียนการแก้ไข**:
  - สร้าง SQL migration [20261011000000_create_uploads_storage_bucket.sql](file:///d:/product_checker/check-products/taxonomy-app/supabase/migrations/20261011000000_create_uploads_storage_bucket.sql) เพื่อ insert แถวลงใน `storage.buckets`
  - กำหนด RLS policies บนตาราง `storage.objects` ให้ครอบคลุมทั้งการ SELECT/INSERT/DELETE ของ `anon` และ `service_role`

---

## 5. การป้องกันข้อผิดพลาดเงียบ (Silent Failures) และการ Sanitize Error 500
* **ปัญหาดั้งเดิม**:
  - Supabase client `.update()` หรือ `.delete()` เมื่อไม่พบแถวที่ตรงกับเงื่อนไข จะคืน HTTP 200/204 โดยไม่มีข้อความ error ทำให้ฝั่ง frontend หลงเชื่อว่าการทำงานสำเร็จและแสดง success toast
  - FastAPI routers เดิม ส่งข้อความ exception ดิบออกทาง `{detail: str(e)}` เมื่อเกิด HTTP 500 ทำให้ stack trace และข้อมูลภายในระบบรั่วไหล
* **บทเรียนการแก้ไข**:
  - ทุกจุดที่มีการ `.update()` ใน Next.js Route Handlers ต้องต่อท้ายด้วย `.select('id')` และตรวจว่ามีแถวถูกอัปเดตจริง ถ้าผลลัพธ์เป็น array ว่าง ให้ตอบกลับ HTTP 404 หรือแจ้ง error ชัดเจน
  - ใน FastAPI เมื่อดักจับ `Exception as e` ให้ใช้ `logger.exception("...")` เพื่อเก็บบันทึกบนเซิร์ฟเวอร์ และส่ง `{detail: "An internal server error occurred while processing..."}` ที่เป็น generic detail กลับไปยัง client เสมอ
