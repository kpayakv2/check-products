# ทะเบียนข้อค้นพบ

**Parent:** [แผนที่](map.md) · เมทริกซ์: [`matrix.md`](matrix.md)

รูปแบบต่อข้อ: ระดับ · มิติ · ชั้น/แถวใน matrix · ป้ายการยืนยัน (`ยิงจริงแล้ว` / `อ่านโค้ดอย่างเดียว`) · ไฟล์:บรรทัด · หลักฐาน · คำตัดสินที่เสนอ + เหตุผล · คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 09/10)

---

## F-001 · P0 · อ่านฐานข้อมูลทั้งหมดได้จากอินเทอร์เน็ตผ่าน `/api/supabase/*` โดยไม่ต้อง unlock

- **มิติ:** 4 (สิทธิ์เข้าถึง)
- **ชั้น:** 2 (Next.js — rewrite) + 4 (PostgREST)
- **การยืนยัน:** `ยิงจริงแล้ว` (GET เข้า `localhost:3000` ไม่มี cookie — เซิร์ฟเวอร์เดียวกับที่ tunnel ส่งเข้า) · ยังไม่ได้ยิงผ่านโดเมนสาธารณะ และยังไม่ได้ตรวจว่ามี Cloudflare Access กั้นหน้า hostname หรือไม่
- **ที่มา:**
  - `taxonomy-app/middleware.ts:32` — GET/HEAD/OPTIONS ทุกเส้นใต้ `/api/*` ผ่านโดยไม่เช็ก cookie ยกเว้น 3 path ใน `GATED_READ_PATHS` (บรรทัด 18)
  - `taxonomy-app/next.config.js:22-23` — rewrite `/api/supabase/:path*` ตรงเข้า Supabase gateway
  - `taxonomy-app/.env.local` — ใช้ key ค่าตั้งต้นสาธารณะของ Supabase CLI (นับเจอ 3 จุด ไม่ได้พิมพ์ค่า)
  - `~/.cloudflared/config.yml` — tunnel `product-checker` ส่ง `check.smilesboo.com` + `ckeck.smilesboo.com` → `localhost:3000` (สร้าง 2026-09-22, กำลังรันอยู่ ณ 2026-10-11)
- **หลักฐาน (2026-10-11, พิมพ์แค่จำนวนแถว):**

  | key ที่ส่ง | ตาราง | ผล |
  |---|---|---|
  | ไม่ส่งเลย | `products` | HTTP 206, `0-0/3508` |
  | ไม่ส่งเลย | `system_settings` | HTTP 200, `*/0` (RLS กั้น) |
  | service_role ค่าตั้งต้น | `system_settings` | HTTP 206, `0-0/4` |
  | service_role ค่าตั้งต้น | `audit_logs` | HTTP 206, `0-0/6937` |

- **ขอบเขตความเสียหาย:** อ่านอย่างเดียว — non-GET ถูก `middleware.ts` กั้น (`อ่านโค้ดอย่างเดียว` ไม่ได้ยิงทดสอบเขียน) ไม่มี log ฝั่งเราให้ตรวจย้อนว่าตั้งแต่ 22 ก.ย. มีใครอ่านไปแล้วหรือไม่
- **ผู้ใช้ตัดสิน (2026-10-11):** **เลื่อนการแก้โค้ดไปตั๋ว 09** โดยระหว่างรอให้ปิด tunnel ไว้ (ทางเลือก ก) — ⚠️ **ยังไม่ได้ปิด**: คำสั่งหยุด `cloudflared` ถูก permission ของ Claude Code ปฏิเสธ รอผู้ใช้หยุดเอง
- **คำตัดสินที่เสนอสำหรับตั๋ว 09:** แก้ — ให้ `middleware.ts` ขอ cookie กับทุก method ใต้ `/api/*` (เว้น `/api/unlock`, `/api/lock`) + เปลี่ยน JWT secret ของ Supabase ออกจากค่าตั้งต้น + พิจารณา Cloudflare Access ถ้าจะเปิด tunnel ใช้งานจริง
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 09, 2026-10-11):** **ปล่อยไว้ (เก็บไว้แก้ทีหลัง)** — ผู้ใช้ตัดสินใจคงสถานะเดิมไว้ก่อน ไม่จัดเข้าชุดงานแก้เร่งด่วนของแผนที่นี้ แต่ต้องระมัดระวังเรื่องการเปิด Cloudflare tunnel สู่ภายนอก
- **ผลต่อเอกสาร:** AGENTS.md หัวข้อ Authentication บรรยาย threat model ว่า "office LAN" — ผิดตั้งแต่มี tunnel (ป้อนตั๋ว 07)

## F-002 · P0 · FastAPI เขียนข้อมูล/สั่งงานหนักได้จากทุกเครื่องในเครือข่ายเดียวกัน ข้าม `middleware.ts`

- **มิติ:** 4 (สิทธิ์เข้าถึง)
- **ชั้น:** 1 (FastAPI)
- **การยืนยัน:** `ยืนยันจากระบบจริง` (ตรวจการ bind, process, firewall rule และ network profile ด้วย PowerShell จริงในตั๋ว 03) · ไม่ได้ยิงเส้นเขียนข้อมูล
- **ที่มา:**
  - `src/api/api_server.py:133` — `host="0.0.0.0"` · บรรทัด 56-58 — CORS `allow_origins=["*"]` · ไม่มี auth dependency ในทุก router
  - `src/api/routers/learn.py:11` `POST /api/v1/learn/verify` (เขียน `keyword_rules` ลง Supabase ตรง) · บรรทัด 37 `POST /api/v1/learn/retrain` (เทรนโมเดลใหม่) · `src/api/routers/system.py:52` `POST /api/v1/upload/csv` (เขียนไฟล์ลงดิสก์) · `src/api/routers/internal_match.py:437` `POST /api/v1/match/scan-internal` (สแกนทั้ง DB แบบ pairwise O(N^2))
  - กระบวนการ: PID 22936 (`python.exe` 3.14 ที่ `C:\Users\minds\AppData\Local\Python\pythoncore-3.14-64\python.exe`) กำลัง Listen บน `0.0.0.0:8000`
  - Windows Firewall: มี Inbound Rule ชื่อ `TCP Query User{E35E434B-6799-40DA-AD02-B015270B8AF8}...` Action **Allow**, Profile **Public**, Protocol **TCP**, LocalPort **Any**
  - Network Profile: การเชื่อมต่อ Wi-Fi (`true_home2G_23d 2`) อยู่ในหมวด **Public**
- **ขอบเขต:** อุปกรณ์ใดๆ ในเครือข่าย Wi-Fi/LAN เดียวกันสามารถยิงตรงเข้า `:8000` เพื่อเขียน DB, สั่งเทรน หรืออัปโหลดไฟล์ได้โดยไม่ต้องผ่าน Next.js `middleware.ts`
- **คำตัดสินที่เสนอสำหรับตั๋ว 09:** แก้ — เปลี่ยนการ bind เป็น `127.0.0.1` ใน `api_server.py` (Next.js คุยผ่าน server-side `FASTAPI_URL` อยู่แล้ว) และ/หรือตรวจ `INTERNAL_API_SECRET` ใน header
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 09, 2026-10-11):** **ปล่อยโค้ดไว้ / แก้กฎข้อ 7 ใน AGENTS.md** — ผู้ใช้ตั้งใจให้ FastAPI รันบน `0.0.0.0` เพื่อเปิดรับการเชื่อมต่อจากเครื่องใน LAN ตรง จึงไม่แก้โค้ด `api_server.py` แต่ปรับแก้ถ้อยคำกฎเหล็กข้อ 7 ใน `AGENTS.md` ให้บันทึกว่าอนุญาตให้ bind `0.0.0.0` บน LAN ได้ และการเชื่อมต่อในเครื่องให้ใช้ `127.0.0.1` ไม่ใช้ `localhost` (ลดระดับความรุนแรงจาก P0 เป็นงานปรับแก้เอกสารในตั๋ว 11)

## F-003 · P1 · `POST /api/v1/match/batch` จับ `HTTPException(400)` แล้ว re-wrap เป็น 500

- **มิติ:** 3 (ตรวจ input) · 5 (สัญญา error)
- **ชั้น:** 1 (FastAPI)
- **การยืนยัน:** `อ่านโค้ดอย่างเดียว`
- **ไฟล์:บรรทัด:** `src/api/routers/match.py:78-82, 121-123`
- **ที่มา:** ใน `match_batch_products` มีการตรวจความยาว `len(request.query_products) > max_batch_size` แล้ว `raise HTTPException(status_code=400, detail=...)` แต่บล็อกฟังก์ชันถูกครอบด้วย `except Exception as e: raise HTTPException(status_code=500, detail=f"Failed to start batch job: {str(e)}")` โดยไม่มี `except HTTPException: raise` นำหน้า เนื่องจาก `HTTPException` สืบทอดจาก `Exception` ข้อผิดพลาด 400 จาก input จึงถูกแปลงเป็น 500 เสมอ ผิดเกณฑ์มิติ 3 ("ผิดแล้วคืน 4xx ไม่ตกไป 500")
- **คำตัดสินที่เสนอ:** แก้ — ใส่ `except HTTPException: raise` ก่อน `except Exception as e:` (เหมือนแพทเทิร์นใน `learn.py` และ `classify.py`) หรือพิจารณาลบ endpoint นี้เนื่องจากเป็นเส้นตาย (ดู F-008)
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 09, 2026-10-11):** **ลบ (Delete)** — ลบ endpoint `POST /api/v1/match/batch` ออกจาก `src/api/routers/match.py` ในตั๋ว 11 เนื่องจากเป็น dead code ที่ไม่มีผู้เรียกในระบบจริง

## F-004 · P1 · `POST /api/v1/upload/csv` ขาดการจำกัดขนาดไฟล์, เสี่ยง path traversal, และไม่มี background task ตามที่ประกาศ

- **มิติ:** 1 (เอกสาร vs โค้ด) · 3 (ตรวจ input) · 4 (สิทธิ์เข้าถึง)
- **ชั้น:** 1 (FastAPI)
- **การยืนยัน:** `อ่านโค้ดอย่างเดียว`
- **ไฟล์:บรรทัด:** `src/api/routers/system.py:53-95`
- **ที่มา:**
  - รับ `UploadFile` แล้วเขียนลงดิสก์ทันทีด้วย `shutil.copyfileobj` โดยไม่ตรวจขนาดไฟล์ แม้ใน `APIConfig` จะมี `max_file_size` กำหนดไว้ (เสี่ยง Disk Exhaustion / DoS)
  - นำ `file.filename` มาต่อ path ตรงๆ (`f"{upload_id}_{file.filename}"`) โดยไม่ผ่านการ sanitize ป้องกัน path traversal
  - Docstring อ้างว่า "for batch processing" และรับ `BackgroundTasks` ในพารามิเตอร์ แต่โค้ดไม่ได้เพิ่มงานเบื้องหลังใดๆ
  - เรียก `pd.read_csv` โดยไม่ระบุ encoding ภาษาไทย ทำให้เกิด `UnicodeDecodeError` แล้วตกไปเป็น 500
- **คำตัดสินที่เสนอ:** ลบ — เนื่องจากเป็นเส้นตายที่ไม่มีผู้เรียกในแอป (ดู F-008) หากต้องการเก็บไว้ ต้องเพิ่มการตรวจขนาดไฟล์และ sanitize filename
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 09, 2026-10-11):** **ลบ (Delete)** — ลบ endpoint `POST /api/v1/upload/csv` ออกจาก `src/api/routers/system.py` ในตั๋ว 11 เนื่องจากเป็น dead code ที่ไม่มีผู้เรียกในระบบจริง

## F-005 · P1 · `POST /api/v1/match/single` กลายพันธุ์ singleton config แบบ in-place ทำให้เกิด race condition

- **มิติ:** 3 (ตรวจ input) · 5 (สัญญา error)
- **ชั้น:** 1 (FastAPI)
- **การยืนยัน:** `อ่านโค้ดอย่างเดียว`
- **ไฟล์:บรรทัด:** `src/api/routers/match.py:24-27`
- **ที่มา:** ฟังก์ชัน `match_single_product` เป็น async แต่สั่งแก้ attribute บน singleton pipeline โดยตรง:
  `pipeline.product_matcher.config.similarity_threshold = request.threshold`
  `pipeline.product_matcher.config.top_k = request.top_k`
  หากมี 2 request เข้ามาพร้อมกันด้วย threshold ต่างกัน จะเกิด race condition เขียนทับค่าคอนฟิกของกันและกัน
- **คำตัดสินที่เสนอ:** พิจารณาลบ (เป็นเส้นตายตาม F-008) หรือส่ง threshold/top_k เป็น parameter รายการเรียกแทนการ mutate singleton config
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 09, 2026-10-11):** **ลบ (Delete)** — ลบ endpoint `POST /api/v1/match/single` ออกจาก `src/api/routers/match.py` ในตั๋ว 11 เนื่องจากเป็น dead code ที่ไม่มีผู้เรียกในระบบจริง

## F-006 · P2 · `GET /` โฆษณา endpoint `/web` ที่ไม่มีอยู่จริง (คืน 404) และมีรายการ endpoint ไม่ครบถ้วน

- **มิติ:** 1 (เอกสาร vs โค้ด)
- **ชั้น:** 1 (FastAPI)
- **การยืนยัน:** `ยิงจริงแล้ว` (GET เข้า `http://127.0.0.1:8000/` ได้ `{"endpoints": {"web_interface": "/web"}}` แต่ GET `http://127.0.0.1:8000/web` ได้ HTTP 404 เนื่องจากโฟลเดอร์ `src/api/web` ไม่มีอยู่จริงบนดิสก์)
- **ไฟล์:บรรทัด:** `src/api/api_server.py:65-67, 86-100`
- **ที่มา:** โค้ดตรวจสอบ `if web_dir.exists():` ก่อน mount แต่ในตัว endpoint `GET /` กลับ hardcode ส่ง `/web` คืนเสมอ และในรายการ endpoints ระบุเพียง 5 จาก 24 endpoint
- **คำตัดสินที่เสนอ:** แก้ — ลบ `/web` ออกจาก response หรืออัปเดต docstring/endpoints ให้ตรงกับความเป็นจริง
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 10, 2026-10-11):** **แก้ (Fix)** — ปรับปรุง response ของ `GET /` ใน `api_server.py` ลบ `/web` ที่คืน 404 ออก และแสดงรายการ endpoints ที่ใช้งานจริงในตั๋ว 11

## F-007 · P2 · สัญญา Error 500 รั่วไหล `str(e)` ออกทาง `{detail}` ใน 14 จาก 24 endpoint ของ FastAPI

- **มิติ:** 5 (สัญญา error)
- **ชั้น:** 1 (FastAPI)
- **การยืนยัน:** `อ่านโค้ดอย่างเดียว`
- **ไฟล์:บรรทัด:**
  - `src/api/routers/system.py:95, 123`
  - `src/api/routers/match.py:71, 123`
  - `src/api/routers/embed.py:35, 59`
  - `src/api/routers/learn.py:35, 62, 131, 151`
  - `src/api/routers/internal_match.py:497, 502, 680, 851`
  - `src/api/routers/classify.py:290, 342`
- **ที่มา:** มีแพทเทิร์น `except Exception as e: raise HTTPException(status_code=500, detail=f"...: {str(e)}")` หรือ `detail=str(e)` อยู่ทั่วไปในเกือบทุก router ทำให้ internal error message, database error หรือ file paths ถูกส่งกลับไปยัง client ขัดกับเกณฑ์มิติ 5
- **คำตัดสินที่เสนอ:** แก้ — เปลี่ยนเป็น log รายละเอียดลง server log และคืน generic detail เช่น `"Internal server error"` ให้กับ client
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 10, 2026-10-11):** **แก้ (Fix)** — ปรับปรุง error handling ใน 14 endpoints ของ FastAPI ไม่ให้ส่ง `str(e)` ดิบออกทาง detail เมื่อเกิด 500 โดยใช้ logging แทนในตั๋ว 11

## F-008 · P3 · ชุด endpoint Legacy Matching & Config (9 endpoint) เป็นเส้นตาย ไม่มีผู้เรียกจาก UI

- **มิติ:** 6 (เส้นตาย/ซ้ำซ้อน)
- **ชั้น:** 1 (FastAPI)
- **การยืนยัน:** `อ่านโค้ดอย่างเดียว` (กวาดหาใน `taxonomy-app/` ไม่พบผู้เรียก)
- **ไฟล์:บรรทัด:**
  - L1-01 `GET /` (`api_server.py:86`)
  - L1-03 `GET /api/v1/config` (`system.py:35`)
  - L1-04 `POST /api/v1/config` (`system.py:40`)
  - L1-05 `POST /api/v1/upload/csv` (`system.py:52`)
  - L1-07 `POST /api/v1/match/single` (`match.py:16`)
  - L1-08 `POST /api/v1/match/batch` (`match.py:73`)
  - L1-11 `GET /api/v1/jobs/{job_id}` (`jobs.py:15`)
  - L1-12 `GET /api/v1/jobs` (`jobs.py:23`)
  - L1-13 `GET /api/v1/results/{job_id}` (`jobs.py:28`)
- **ที่มา:** โค้ดชุด Phase 5 เดิมถูกแทนที่ด้วย `internal_match.py` (`scan-internal`, `file-dedup`, `import-dedup`) และ Edge Functions ไปแล้ว แต่ยังคงค้างอยู่ใน FastAPI ทำให้เกิด code bloat และหนี้ทางเทคนิค
- **คำตัดสินที่เสนอ:** ปล่อยไว้ก่อน แล้วตัดสินใจลบในตั๋ว 10/11
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 10, 2026-10-11):** **ลบ (Delete)** — ลบ 8 legacy dead endpoints ใน `match.py`, `system.py`, และ `jobs.py` ออกทั้งหมดในตั๋ว 11

## F-009 · P3 · ความไม่สม่ำเสมอเชิงรูปแบบของ payload (camelCase ใน response model, ซ้อน envelope ไม่ตรงแบบ)

- **มิติ:** 7 (ความสม่ำเสมอเชิงรูปแบบ)
- **ชั้น:** 1 (FastAPI)
- **การยืนยัน:** `อ่านโค้ดอย่างเดียว`
- **ไฟล์:บรรทัด:**
  - `src/api/models.py:169-206` — `ProductMatchResult` และ `FileDeduplicationStats` ใช้ฟิลด์ camelCase (`newProduct`, `oldProduct`, `oldProductId`, `oldPrice`, `mlPrediction`, `identityVerdict`, `totalNewProducts`, etc.) ซึ่งถูกส่งออกทาง L1-20 (`/api/v1/match/file-dedup`) และ L1-21 (`/api/v1/match/import-dedup`)
  - `src/api/routers/learn.py:52, 145` — L1-15 และ L1-17 คืน envelope `{status: "success", data/history: ...}` ซึ่งเป็นรูปแบบของ Next.js API มากกว่า FastAPI
  - L1-01, L1-04, L1-05, L1-14 คืน raw dict โดยไม่มี Pydantic `response_model`
- **คำตัดสินที่เสนอ:** ปรับในตั๋ว 08 (ถ้ามีแหล่งความจริงของ type เดียว) หรือบันทึกเป็นข้อยกเว้นทางรูปแบบในตั๋ว 10
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 10, 2026-10-11):** **แก้ (Fix)** — ปรับ casing และ envelope ของ models ใน FastAPI ให้เป็น `snake_case` สม่ำเสมอในตั๋ว 11

## F-010 · P3 · ขาด Automated Tests ชี้ตรงถึง 8 endpoint ในชั้น FastAPI

- **มิติ:** 8 (ความครอบคลุมของเทสต์)
- **ชั้น:** 1 (FastAPI)
- **การยืนยัน:** `อ่านโค้ดอย่างเดียว` (ค้นหา endpoint path ใน `tests/`)
- **รายการที่ขาด:**
  - L1-01 `GET /`
  - L1-03 `GET /api/v1/config`
  - L1-04 `POST /api/v1/config`
  - L1-05 `POST /api/v1/upload/csv`
  - L1-12 `GET /api/v1/jobs`
  - L1-13 `GET /api/v1/results/{job_id}`
  - L1-20 `POST /api/v1/match/file-dedup`
  - L1-21 `POST /api/v1/match/import-dedup`
- **ที่มา:** 6 เส้นแรกเป็นเส้นตายที่ไม่มีเทสต์ ส่วน 2 เส้นหลัง (`file-dedup` และ `import-dedup`) มีการใช้งานจริงในระบบแต่ยังไม่มี integration test ชี้ตรงใน pytest
- **คำตัดสินที่เสนอ:** สำหรับเส้นตายให้พิจารณาลบ ส่วนเส้นที่ใช้งานจริงให้สร้าง integration test ในตั๋วแก้ (ตั๋ว 11)
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 10, 2026-10-11):** **แก้/เขียนเทสต์เฉพาะเส้นจริง** — เขียน integration test สำหรับเส้นที่ใช้งานจริง (`POST /api/v1/learn/verify`, `POST /api/v1/match/scan-internal`) และข้ามเส้นตายที่ถูกลบในตั๋ว 11

## F-011 · P1 · `GET /api/recheck` อ่านข้อมูลผ่าน `supabaseAdmin` แต่ตกหล่นจาก `GATED_READ_PATHS` ของ `middleware.ts`

- **มิติ:** 4 (สิทธิ์เข้าถึง)
- **ชั้น:** 2 (Next.js)
- **การยืนยัน:** `ยิงจริงแล้ว` (GET เข้า `http://localhost:3000/api/recheck` โดยไม่มี session cookie ได้ HTTP 200 พร้อมข้อมูล `product_category_suggestions`, `products` และ metadata รายการที่รอการตรวจซ้ำทั้งหมด)
- **ไฟล์:บรรทัด:** `taxonomy-app/middleware.ts:18, 32`, `taxonomy-app/app/api/recheck/route.ts:3, 33-55`
- **ที่มา:** ใน `middleware.ts:18` ระบุ `const GATED_READ_PATHS = new Set(['/api/settings', '/api/import/history', '/api/pos-export'])` แต่ไม่ได้ใส่ `'/api/recheck'` เข้าไป ทั้งที่ route นี้เป็น service-role read (ใช้ `supabaseAdmin`) ที่ bypass RLS ส่งผลให้คำขอ GET อ่านข้อมูลที่ยังไม่ได้อนุมัติได้โดยตรงจากภายนอกโดยไม่ต้อง unlock
- **คำตัดสินที่เสนอ:** แก้ — เพิ่ม `'/api/recheck'` ลงใน `GATED_READ_PATHS` ของ `middleware.ts` (หรือครอบคลุมในตั๋ว 09 เมื่อตัดสินใจ gate ทุก GET ใต้ `/api/*`)
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 09, 2026-10-11):** **แก้ (Fix)** — เพิ่ม `'/api/recheck'` ลงใน `GATED_READ_PATHS` ของ `taxonomy-app/middleware.ts:18` ในตั๋ว 11 เพื่อบังคับตรวจ session cookie เสมอ ป้องกันการอ่าน unreviewed products โดยไม่ได้รับอนุญาต

## F-012 · P1 · `POST /api/similarity/resolve` ไม่ตรวจว่ามีแถวถูกอัปเดตจริงใน `products` หรือไม่ เสี่ยงเกิด false success

- **มิติ:** 5 (สัญญา error)
- **ชั้น:** 2 (Next.js)
- **การยืนยัน:** `อ่านโค้ดอย่างเดียว`
- **ไฟล์:บรรทัด:** `taxonomy-app/app/api/similarity/resolve/route.ts:62-73`
- **ที่มา:** โค้ดสั่ง `.from('products').update({ status: 'duplicate' }).eq('id', remove_id)` โดยไม่มี `.select()` และไม่ตรวจจำนวนแถวที่ได้รับผลกระทบ หาก client ส่ง `remove_id` ที่ไม่มีอยู่จริง Supabase จะคืน `error: null` และไม่มีแถวใดถูกแก้ไข แต่ route ยังคงทำงานต่อ (upsert `similarity_matches`) และคืน HTTP 200 `{ success: true }` ทำให้ UI แสดง toast สำเร็จปลอม ขัดกับเกณฑ์มิติ 5 ข้อ (ค)
- **คำตัดสินที่เสนอ:** แก้ — เปลี่ยนไปใช้ `updateRow('products', remove_id, ...)` จาก `utils/admin-db.ts` หรือเติม `.select('id')` และตรวจว่ามีข้อมูลคืนมาหรือไม่ หากไม่มีให้คืน HTTP 404
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 09, 2026-10-11):** **แก้ (Fix)** — แก้ `taxonomy-app/app/api/similarity/resolve/route.ts` ให้ตรวจแถวที่ถูกอัปเดตจริงผ่าน `.select('id')` หรือ `updateRow` ในตั๋ว 11 เพื่อป้องกัน false success เมื่อ `remove_id` ไม่มีอยู่จริง

## F-013 · P2 · 6 route ในชั้น Next.js ขาดการตรวจสอบ input ด้วย Zod (ใช้ `request.json()` ดิบ หรือตรวจไม่ครบ) และ crash เป็น 500 เมื่อ UUID ผิด

- **มิติ:** 3 (ตรวจ input) · 5 (สัญญา error)
- **ชั้น:** 2 (Next.js)
- **การยืนยัน:** `ยิงจริงแล้ว` (GET `/api/similarity/dummy-id` ได้ 500 แทนที่จะได้ 400/404) · `อ่านโค้ดอย่างเดียว` บนเส้นเขียน
- **ไฟล์:บรรทัด:**
  - `app/api/synonyms/[id]/route.ts:15` (PUT รับ `request.json()` ดิบแล้วส่งเข้า `updateRow`)
  - `app/api/taxonomy/[id]/route.ts:17` (PUT รับ `request.json()` ดิบแล้วส่งเข้า `updateRow`)
  - `app/api/settings/route.ts:21` (PUT รับ `request.json()` ดิบแล้วส่งเข้า `writeSettingsRow`)
  - `app/api/synonyms/route.ts:33` (POST ตรวจมือแค่ชื่อและ terms ไม่ตรวจ schema ของ terms)
  - `app/api/similarity/[id]/route.ts:9, 26` (GET/POST ไม่ตรวจ UUID param ทำให้ query DB ล้มเหลวเป็น 500, POST ตรวจ body แค่ 2 ฟิลด์)
  - `app/api/products/route.ts:23` (POST ตรวจมือแค่ name_th นอกนั้นยัดเข้า DB ทั้งหมด)
- **ที่มา:** ขาด schema validation ที่ขอบระบบ ทำให้ข้อมูลผิดพลาดไหลไปถึง PostgreSQL และล้มเหลวด้วย exception ภายในกลายเป็น 500 แทนที่จะตัดจบด้วย 400 ที่ boundary
- **คำตัดสินที่เสนอ:** แก้ — สร้าง Zod schema ให้กับ body/params ของทุก route ที่ใช้งานจริง และลบ route ที่เป็นเส้นตาย
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 10, 2026-10-11):** **แก้ (Fix)** — เพิ่ม Zod validation ให้กับ route ที่ใช้งานจริงใน Next.js (`synonyms/[id]`, `taxonomy/[id]`, `settings`, `synonyms`, `products`) ในตั๋ว 11 เพื่อป้องกัน server crash 500 เมื่อ input ผิดพลาด

## F-014 · P2 · Route ที่ตกค้าง 6 เส้นใช้งาน `DatabaseService` (anon key) ซึ่งขัดกับโมเดล RLS ของระบบและล้มเหลวเงียบๆ หรือพังเป็น 500

- **มิติ:** 4 (สิทธิ์เข้าถึง) · 5 (สัญญา error)
- **ชั้น:** 2 (Next.js)
- **การยืนยัน:** `อ่านโค้ดอย่างเดียว`
- **ไฟล์:บรรทัด:**
  - L2-12, L2-13: `app/api/similarity/[id]/route.ts` (เรียก `getSimilarityMatches` / `createSimilarityMatch` ผ่าน anon key ซึ่ง RLS ไม่อนุญาตให้ insert)
  - L2-14: `app/api/taxonomy/route.ts:118, 127` (GET เรียกผ่าน anon key)
  - L2-16: `app/api/synonyms/route.ts:11` (GET เรียกผ่าน anon key)
  - L2-19, L2-20: `app/api/products/route.ts:10, 44, 60` (GET/POST เรียกผ่าน anon key โดยการเขียน `product_attributes` จะล้มเหลวเพราะ RLS)
- **ที่มา:** Route ยุคแรกสร้างโดยอาศัย `DatabaseService` ที่ผูกกับ anon key ซึ่งในโมเดลความปลอดภัยปัจจุบัน table เหล่านี้มี RLS บล็อกการทำงานของ anon key ไว้
- **คำตัดสินที่เสนอ:** ลบ — เนื่องจากทั้งหมดเป็นเส้นตายที่ไม่มีผู้เรียกในแอป (ดู F-015)
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 10, 2026-10-11):** **แก้/ลบ** — ลบเส้นทางที่เป็น dead routes (ตาม F-015) และหากมีเส้นที่ต้องคงไว้ให้ปรับใช้ `supabaseAdmin` แทนการใช้ `DatabaseService` anon key ในตั๋ว 11

## F-015 · P3 · 6 endpoint ในชั้น Next.js เป็นเส้นตาย ไม่มีผู้เรียกจากหน้าเว็บใดๆ

- **มิติ:** 6 (เส้นตาย/ซ้ำซ้อน)
- **ชั้น:** 2 (Next.js)
- **การยืนยัน:** `อ่านโค้ดอย่างเดียว` (กวาดหาในโค้ด components และ pages ของ `taxonomy-app`)
- **รายการ:**
  - L2-12 `GET /api/similarity/[id]`
  - L2-13 `POST /api/similarity/[id]`
  - L2-14 `GET /api/taxonomy` (หน้า taxonomy ใช้ `DatabaseService.getTaxonomyTree` ฝั่ง client ตรง)
  - L2-16 `GET /api/synonyms` (SynonymsPanel ใช้ `DatabaseService.getSynonyms` ฝั่ง client ตรง)
  - L2-19 `GET /api/products`
  - L2-20 `POST /api/products`
- **คำตัดสินที่เสนอ:** ลบในตั๋ว 10/11 เพื่อลดความซ้ำซ้อนและภาระในการบำรุงรักษา
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 10, 2026-10-11):** **ลบ (Delete)** — ลบ 6 dead routes ใน Next.js (`/api/similarity/[id]` GET/POST, `/api/taxonomy` GET, `/api/synonyms` GET, `/api/products` GET/POST) ออกในตั๋ว 11

## F-016 · P3 · Envelope ของ Response ใน Next.js แตกต่างกัน 5 รูปแบบ (ข้อมูลเสริมอยู่นอกฟิลด์ `data`)

- **มิติ:** 7 (ความสม่ำเสมอเชิงรูปแบบ)
- **ชั้น:** 2 (Next.js)
- **การยืนยัน:** `อ่านโค้ดอย่างเดียว`
- **ไฟล์:บรรทัด:**
  - L2-02 `GET /api/pos-export` คืน `{ success: true, total, issues, preview }`
  - L2-08 `POST /api/verify` และ L2-23 `POST /api/recheck` คืน `{ success: true, action, category_id, learned }`
  - L2-14 `GET /api/taxonomy` คืน `{ success: true, data, meta: { count, limit, offset } }`
  - L2-21 `POST /api/import/commit` คืน `{ success: true, updated, suggestions }`
  - L2-22 `GET /api/recheck` คืน `{ success: true, data, total, limit, offset }`
- **ที่มา:** เกณฑ์มิติ 7 กำหนดให้ฟิลด์ข้อมูลที่ส่งกลับต้องอยู่ใน `data` หรือ `meta` โดย payload หลักต้องเป็น `{ success, data?, error? }`
- **คำตัดสินที่เสนอ:** ปรับ envelope ให้สม่ำเสมอในตั๋ว 10
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 10, 2026-10-11):** **แก้ (Fix)** — ปรับ response envelope ของ Next.js API routes ให้เป็นรูปแบบมาตรฐาน `{ success: true, data: ..., error?: ... }` ในตั๋ว 11

## F-017 · P3 · ขาด Automated Tests ชี้ตรงถึง 4 endpoint ในชั้น Next.js

- **มิติ:** 8 (ความครอบคลุมของเทสต์)
- **ชั้น:** 2 (Next.js)
- **การยืนยัน:** `อ่านโค้ดอย่างเดียว` (ค้นหาใน `taxonomy-app/__tests__` และ `taxonomy-app/e2e`)
- **รายการที่ขาด:**
  - L2-01 `POST /api/similarity/resolve`
  - L2-12 `GET /api/similarity/[id]`
  - L2-13 `POST /api/similarity/[id]`
  - L2-25 `DELETE /api/keyword-rules/[id]`
- **คำตัดสินที่เสนอ:** ลบเส้นตาย (L2-12, L2-13) และเพิ่ม integration/unit test สำหรับเส้นที่ใช้งานจริง (L2-01, L2-25) ในตั๋ว 11
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 10, 2026-10-11):** **แก้/เขียนเทสต์เฉพาะเส้นจริง** — เขียน integration test สำหรับ Next.js routes ที่ใช้งานจริง (`POST /api/similarity/resolve`, `DELETE /api/keyword-rules/[id]`) และข้ามเส้นตายที่ถูกลบในตั๋ว 11

## F-018 · P1 · Edge Function `generate-embeddings-local` มี schema สองด้านไม่ตรงกับ FastAPI (ส่ง `{texts: [text]}` แต่ router รับ `{text}`, คาดหวัง `data.embeddings[0]` แต่ router คืน `embedding`)

- **มิติ:** 1 (เอกสาร vs โค้ด) · 5 (สัญญา error)
- **ชั้น:** 3 (Supabase Edge Functions)
- **การยืนยัน:** `อ่านโค้ดอย่างเดียว`
- **ไฟล์:บรรทัด:** `taxonomy-app/supabase/functions/generate-embeddings-local/index.ts:25-39`, `src/api/routers/embed.py:11-31`, `src/api/models.py:101-116`
- **ที่มา:** Edge function ส่ง `{ texts: [text] }` ไปยัง `http://127.0.0.1:8000/api/embed` (L1-09) ซึ่ง Pydantic model `EmbeddingRequest` บังคับฟิลด์ `text: str` ส่งผลให้ FastAPI ตีกลับด้วย HTTP 422 Unprocessable Entity ทันที และบรรทัด 39 อ้างอิง `data.embeddings[0]` ซึ่งใน `EmbeddingResponse` ของ FastAPI มีเฉพาะฟิลด์ `embedding` (singular list of floats) หากฟังก์ชันนี้ถูกเรียกใช้งานจะพัง 100%
- **คำตัดสินที่เสนอ:** ลบ — เนื่องจากเป็นเส้นตายที่ไม่มีผู้เรียกในแอป (ดู F-019)
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 09, 2026-10-11):** **ลบ (Delete)** — ลบโฟลเดอร์ Edge Functions `generate-embeddings-local` ทิ้งในตั๋ว 11 เนื่องจากเป็น dead code ที่ไม่มีผู้เรียกในระบบจริง

## F-019 · P2 · Edge Functions ทั้ง 3 ตัวใน Supabase เป็นเพียง proxy ส่งต่อเข้า FastAPI บน 127.0.0.1 แต่ไม่ได้ถูกเปิดใช้งานในเครื่อง (Kong คืน 503 name resolution failed) และไม่มีผู้เรียกในแอปพลิเคชันจริง

- **มิติ:** 6 (เส้นตาย/ซ้ำซ้อน)
- **ชั้น:** 3 (Supabase Edge Functions)
- **การยืนยัน:** `ยิงจริงแล้ว` (ยิงผ่าน `:54331/functions/v1/...` ได้ 503 `{"message":"name resolution failed"}` เนื่องจากไม่มี container edge runtime รันอยู่ใน Docker Compose ท้องถิ่น) · `อ่านโค้ดอย่างเดียว` (กวาดหาผู้เรียกในโค้ด frontend และ server ไม่พบการเรียกใช้งานจริง มีเพียง 2 สคริปต์ทดสอบ `test_real_classification.py` และ `test-api.js`)
- **ไฟล์:บรรทัด:** `taxonomy-app/supabase/functions/*`, `taxonomy-app/supabase/config.toml:304-326`
- **ที่มา:** ในอดีตเคยมีแนวคิดจะใช้ Supabase Edge Functions เป็นตัวกลาง แต่ต่อมา Next.js เปลี่ยนไปคุยกับ FastAPI โดยตรงผ่าน Next.js rewrites (`/api/fastapi/*`) Edge Functions ทั้ง 3 ตัวจึงตกค้าง ไม่มีการรัน edge runtime ใน Docker Compose ท้องถิ่น และใน `product-deduplication/index.ts:13, 50` มีพารามิเตอร์ `method` ใน interface แต่โค้ดละเลยไม่ส่งต่อให้ FastAPI
- **คำตัดสินที่เสนอ:** ลบทั้งโฟลเดอร์ `taxonomy-app/supabase/functions/` ในตั๋ว 10/11 เพื่อลดความซ้ำซ้อนและความสับสนทางสถาปัตยกรรม
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 10, 2026-10-11):** **ลบ (Delete)** — ลบโฟลเดอร์ `taxonomy-app/supabase/functions/` ทั้งหมดออกในตั๋ว 11 เพื่อลดความซ้ำซ้อนและความสับสนทางสถาปัตยกรรม

## F-020 · P2 · `WS /ws` เปิดรับการเชื่อมต่อโดยไม่มี authentication ใดๆ และไม่ตรวจชนิด frame (binary frame ทำ connection แครชและ connection ค้างในหน่วยความจำ)

- **มิติ:** 3 (ตรวจ input) · 4 (สิทธิ์เข้าถึง) · 5 (สัญญา error)
- **ชั้น:** 5 (WebSocket)
- **การยืนยัน:** `ยิงจริงแล้ว` (เชื่อมต่อจาก script สำเร็จโดยไม่ต้องยืนยันตัวตน, ส่ง binary frame ได้รับ `ConnectionClosedError` ขาดการ close frame ตามปกติ)
- **ไฟล์:บรรทัด:** `src/api/api_server.py:103-112`, `src/api/custom_websockets.py:5-30`
- **ที่มา:** `api_server.py:108-111` ทำเพียง `while True: await websocket.receive_text()` โดยดักเฉพาะ `WebSocketDisconnect` เมื่อ client ส่ง binary frame จะเกิด `RuntimeError` หลุดจาก handler โดยที่ `manager.disconnect(websocket)` ไม่ถูกเรียก (ไม่มี `finally:`) ทำให้ connection object ตกค้างใน `manager.active_connections` และไม่มีการตรวจ token หรือ origin ใดๆ
- **คำตัดสินที่เสนอ:** ลบ WebSocket endpoint และระบบ broadcast ออกทั้งหมดในตั๋ว 10/11 (ดู F-021)
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 10, 2026-10-11):** **ลบ (Delete)** — ลบ WebSocket endpoint `/ws` และ handler ใน `api_server.py` ออกในตั๋ว 11

## F-021 · P3 · ระบบ WebSocket ฝั่งเซิร์ฟเวอร์ (L5-01 ถึง L5-07) ทั้งหมด 7 รายการเป็นระบบตายที่ไม่มีผู้รับ (frontend ไม่มีโค้ดเชื่อมต่อ WebSocket แม้แต่จุดเดียว)

- **มิติ:** 6 (เส้นตาย/ซ้ำซ้อน)
- **ชั้น:** 5 (WebSocket)
- **การยืนยัน:** `อ่านโค้ดอย่างเดียว` (กวาดหา `WebSocket` / `ws://` ใน `taxonomy-app` ทั้งหมด พบ 0 ผู้รับ มีเพียงคอมเมนต์ใน `utils/api.ts:270`)
- **ไฟล์:บรรทัด:** `src/api/custom_websockets.py:34`, `src/api/api_server.py:103`, `src/api/services/background_jobs.py:57, 96, 110`, `src/api/routers/system.py:46`, `src/api/routers/match.py:60, 107`
- **ที่มา:** ระบบ broadcast ถูกเขียนไว้ในยุคแรกเพื่อส่งความคืบหน้าของ batch matching แต่หน้าบ้าน (`WizardTab`, `DeduplicationStep`, ฯลฯ) เปลี่ยนไปใช้วิธี HTTP request ตรงและ Next.js UI state ทั้งหมด ทำให้โค้ด WebSocket ทั้ง 7 รายการกลายเป็น dead code ส่งข้อความไปหาความว่างเปล่า
- **คำตัดสินที่เสนอ:** ลบโค้ด WebSocket ทั้งหมด (ไฟล์ `custom_websockets.py`, endpoint `/ws`, และการเรียก `notify_websockets`) ในตั๋ว 10/11
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 10, 2026-10-11):** **ลบ (Delete)** — ลบ `src/api/custom_websockets.py` และ event publishers ทั้งหมดใน FastAPI ออกในตั๋ว 11 เนื่องจาก frontend ไม่เคยต่อ WebSocket

## F-022 · P3 · ขาด Automated Tests สำหรับ Edge Functions และ WebSocket ทั้งหมด (10 รายการ)

- **มิติ:** 8 (ความครอบคลุมของเทสต์)
- **ชั้น:** 3 (Supabase Edge Functions) · ชั้น 5 (WebSocket)
- **การยืนยัน:** `อ่านโค้ดอย่างเดียว` (ค้นหาใน `tests/` และ `taxonomy-app/__tests__` ไม่พบเทสต์สำหรับ Edge Functions หรือ `/ws`)
- **รายการ:** L3-01 ถึง L3-03 และ L5-01 ถึง L5-07
- **คำตัดสินที่เสนอ:** เนื่องจากทั้งสองชั้นเป็นเส้นตายที่เสนอให้ลบทั้งหมด จึงไม่จำเป็นต้องเขียนเทสต์เพิ่ม ให้ลบโค้ดออกในตั๋ว 10/11
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 10, 2026-10-11):** **ข้าม/ลบ** — ข้ามการเขียนเทสต์สำหรับ Edge Functions และ WebSocket เนื่องจากระบบถูกลบออกทั้งหมดในตั๋ว 11

## F-023 · P1 · `StorageImport.tsx` อ่านตาราง `imports` ด้วย anon key โดน RLS กรองคืน `[]` เงียบๆ ทำให้หน้าเว็บไม่แสดงรายการนำเข้าจาก Storage

- **มิติ:** 4 (สิทธิ์เข้าถึง) · 5 (สัญญา error)
- **ชั้น:** 4 (PostgREST ตรง)
- **การยืนยัน:** `อ่านโค้ดอย่างเดียว` (ตรวจสอบนโยบาย `imports_read` ใน `pg_policies` พบ qual บังคับ `auth.role() = ANY (ARRAY['taxonomy_editor', 'taxonomy_admin'])` ซึ่ง anon key ไม่ตรงเงื่อนไข)
- **ไฟล์:บรรทัด:** `taxonomy-app/components/Import/StorageImport.tsx:89-98`
- **ที่มา:** ใน `StorageImport.tsx` เรียก `supabase.from('imports').select(...)` โดยตรงด้วย anon key แทนที่จะเรียกผ่าน `/api/import/history` (ซึ่งใช้ `supabaseAdmin`) ส่งผลให้ PostgREST คืน `data: []` และ `error: null` หน้าจอจึงแสดงว่าไม่มีไฟล์ใน Storage ให้เลือกนำเข้าเสมอ ขัดกับเกณฑ์มิติ 4 และมิติ 5
- **คำตัดสินที่เสนอ:** แก้ — เปลี่ยน `StorageImport.tsx` ให้เรียกผ่าน `GET /api/import/history` เหมือน `ImportHistory.tsx` ในตั๋ว 11
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 09, 2026-10-11):** **แก้ (Fix)** — ปรับ `StorageImport.tsx` ให้เปลี่ยนจากการ query PostgREST ตรงด้วย anon key มาใช้ `fetch('/api/import/history')` ในตั๋ว 11 เพื่อแก้ปัญหาหน้าจอคืนค่าว่างเปล่าเงียบๆ

## F-024 · P2 · Storage Bucket `uploads` ไม่มีอยู่ในฐานข้อมูล (ตาราง `storage.buckets` ว่างเปล่า) ทำให้ L4-07 และ L4-08 ล้มเหลวเมื่อกดดาวน์โหลด

- **มิติ:** 1 (เอกสาร vs โค้ด) · 5 (สัญญา error)
- **ชั้น:** 4 (Supabase Storage)
- **การยืนยัน:** `ยิงจริงแล้ว` (query `SELECT COUNT(*) FROM storage.buckets;` ผ่าน MCP `postgres` ได้ผลลัพธ์ 0 แถว)
- **ไฟล์:บรรทัด:** `taxonomy-app/components/Import/StorageImport.tsx:116`, `taxonomy-app/components/Import/ImportHistory.tsx:94`
- **ที่มา:** สคริปต์ migration ในเครื่องไม่ได้สร้าง bucket `uploads` ใน `storage.buckets` เมื่อผู้ใช้กดดาวน์โหลดไฟล์จากหน้า StorageImport หรือ ImportHistory คำสั่ง `supabase.storage.from('uploads').download(...)` จะล้มเหลวด้วยข้อผิดพลาด Bucket not found
- **คำตัดสินที่เสนอ:** แก้ — เพิ่ม migration หรือ seed script สร้าง bucket `uploads` ในเครื่องในตั๋ว 11
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 10, 2026-10-11):** **แก้ (Fix)** — เพิ่ม migration/seed script สร้าง bucket `uploads` ใน `storage.buckets` ในตั๋ว 11 เพื่อให้ดาวน์โหลดไฟล์สำเร็จ

## F-025 · P2 · `DatabaseService` มีเมธอดเขียนข้อมูลด้วย anon key 20 เมธอดที่ขัดกับ RLS (11 เมธอดพังแบบเงียบ คืน 200 `[]` หรือ 204 และ 9 เมธอดถูกบล็อก 403) ทั้งหมดเป็น dead code

- **มิติ:** 4 (สิทธิ์เข้าถึง) · 5 (สัญญา error) · 6 (เส้นตาย/ซ้ำซ้อน)
- **ชั้น:** 4 (DatabaseService)
- **การยืนยัน:** `อ่านโค้ดอย่างเดียว` (ตรวจสอบนโยบาย `pg_policies` ของทั้ง 12 ตาราง)
- **ไฟล์:บรรทัด:** `taxonomy-app/utils/supabase.ts:352-780`
- **รายการ:**
  - UPDATE / DELETE ที่พังเงียบ (10 เมธอด): `updateTaxonomyNode`, `deleteTaxonomyNode`, `updateSynonym`, `deleteSynonym`, `updateImport`, `updateSystemSettings`, `updateRegexRule`, `deleteRegexRule`, `updateKeywordRule`, `deleteKeywordRule`
  - SELECT ที่พังเงียบ (1 เมธอด): `getSystemSettings` (คืน `[]` เนื่องจาก RLS `system_settings_read`)
  - INSERT ที่โดนบล็อก 403 (9 เมธอด): `createSimilarityMatch`, `createProduct`, `createProductAttribute`, `createTaxonomyNode`, `createSynonym` (มีบั๊ก schema ซ้อนหากส่ง terms), `createSynonymTerm`, `createImport`, `createProductCategorySuggestion`, `createRegexRule`, `createKeywordRule`
- **ที่มา:** เมธอดเหล่านี้ถูกเขียนขึ้นในยุคแรกก่อนมี RLS ต่อมาเมื่อระบบเพิ่ม RLS แต่ไม่มีระบบ Login หน้า UI จึงเปลี่ยนไปใช้ Next.js API routes (`/api/*`) ผ่าน `supabaseAdmin` แทน แต่ทิ้งเมธอดเหล่านี้ไว้ใน `DatabaseService` ทำให้เป็นกับดักทางโค้ดที่หากมีใครเผลอเรียกจะล้มเหลวเงียบๆ ทันที
- **คำตัดสินที่เสนอ:** ลบเมธอดเหล่านี้ทั้งหมดออกจาก `DatabaseService` ในตั๋ว 10/11
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 10, 2026-10-11):** **ลบ (Delete)** — ลบ 20 write methods ใน `DatabaseService` ที่ใช้ anon key และเป็น dead code ออกจาก `taxonomy-app/utils/supabase.ts` ในตั๋ว 11

## F-026 · P3 · `DatabaseService` มีเมธอดอ่านข้อมูลที่เป็นเส้นตาย 7 เมธอด ไม่มีผู้เรียกใช้งานจากหน้าเว็บใดๆ

- **มิติ:** 6 (เส้นตาย/ซ้ำซ้อน)
- **ชั้น:** 4 (DatabaseService)
- **การยืนยัน:** `อ่านโค้ดอย่างเดียว`
- **ไฟล์:บรรทัด:** `taxonomy-app/utils/supabase.ts:282, 319, 333, 381, 492, 570, 590`
- **รายการ:** `getTaxonomyNodes` (L4-10), `getProducts` (L4-14), `getSimilarityMatches` (L4-15), `getTaxonomyNode` (L4-23), `getNextSortOrder` (L4-24), `searchTaxonomyNodes` (L4-27), `getSynonymTerms` (L4-33)
- **คำตัดสินที่เสนอ:** ลบออกจาก `DatabaseService` ในตั๋ว 10/11
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 10, 2026-10-11):** **ลบ (Delete)** — ลบ 7 dead read methods ออกจาก `DatabaseService` ใน `taxonomy-app/utils/supabase.ts` ในตั๋ว 11

## F-027 · P3 · Type definitions สำคัญถูกประกาศซ้ำกระจัดกระจาย 11 รายการ ใน 25 ไฟล์ทั่ว frontend ขาด Single Source of Truth

- **มิติ:** 2 (ความสอดคล้องของ type ข้ามรอยต่อ) · 7 (ความสม่ำเสมอเชิงรูปแบบ)
- **ชั้น:** ข้ามชั้น (Frontend Types)
- **การยืนยัน:** `อ่านโค้ดอย่างเดียว` (กวาด regex ทั่ว `taxonomy-app`)
- **รายการที่พบ:**
  1. `TaxonomyNode` (3 ที่: `app/api/taxonomy/route.ts:11`, `components/data-quality/RecheckTab.tsx:23`, `utils/supabase.ts:25`)
  2. `Product` (2 ที่: `components/data-quality/VerifyTab.tsx:24`, `utils/supabase.ts:100`)
  3. `KeywordRule` (3 ที่: `app/settings/page.tsx:42`, `components/data-quality/AutoLearnTab.tsx:17`, `utils/supabase.ts:83`)
  4. `ImportRecord` (2 ที่: `components/Import/ImportHistory.tsx:20`, `components/Import/StorageImport.tsx:23`)
  5. `ExportableProduct` (3 ที่: `app/api/pos-export/route.ts:10`, `utils/pos-export.ts:48`, `__tests__/utils/pos-export.test.ts:12`)
  6. `PosExportRow` (3 ที่: `app/api/pos-export/route.ts:11`, `utils/pos-export.ts:58`, `__tests__/utils/pos-export.test.ts:13`)
  7. `TaxonomyNodeLite` (3 ที่: `app/api/pos-export/route.ts:12`, `utils/pos-export.ts:42`, `__tests__/utils/pos-export.test.ts:14`)
  8. `ItemIdCandidate` (2 ที่: `utils/pos-item-id.ts:42`, `__tests__/utils/pos-item-id.test.ts:11`)
  9. `ApiResponse` (2 ที่: `app/api/taxonomy/route.ts:27`, `__tests__/api/simple-api.test.ts:4`)
  10. `Call` / `Filter` (3-4 ที่ใน `__tests__/api/*`)
  11. ข้ามภาษา Python vs TypeScript: `CategoryRequest`/`CategoryResponse` ใน `classify.py` vs `types/import.ts`, `CleanTextRequest`/`CleanTextResponse` ใน `models.py` vs inline ใน `DataCleaningStep.tsx`
- **คำตัดสินที่เสนอ:** รวบรวมข้อมูลส่งต่อให้ตั๋ว 08 (Single Source of Truth) ตัดสินใจวางสถาปัตยกรรม Type กลาง
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 10, 2026-10-11):** **แก้ (Fix)** — ดำเนินการตาม ADR 0004 ที่ตกลงในตั๋ว 08 (Two-Pillar Type Generation: `supabase gen types` สำหรับ DB, `openapi-typescript` สำหรับ AI, รวมศูนย์ View types ใน `types/`) กำจัด 11 duplicate types ใน 25 ไฟล์ ในตั๋ว 11

## F-028 · P2 · เอกสาร `docs/api/*` มีความขัดแย้งเชิงสถาปัตยกรรม (อ้าง Edge Function First, อ้าง SQLite, ตัวอย่าง payload `/api/v1/clean` ผิดชนิด)

- **มิติ:** 1 (เอกสาร vs โค้ด)
- **ชั้น:** เอกสาร (`docs/api/`)
- **การยืนยัน:** `อ่านโค้ดอย่างเดียว` (เทียบเนื้อหาใน 5 ไฟล์ของ `docs/api/` กับโค้ดจริงใน `src/api` และ `taxonomy-app`)
- **ไฟล์:บรรทัด:** `docs/api/API_ARCHITECTURE.md`, `docs/api/api-reference.md:84-88`, `docs/api/analyze-capabilities.md:21-23`, `docs/api/quick-reference.md`
- **ที่มา:**
  - `docs/api/API_ARCHITECTURE.md` (มี inbound link จาก `docs/INDEX.md`) อ้างว่าระบบยึดหลัก "Edge Function First และเก็บ business logic ทั้งหมดไว้ใน Edge Functions" พร้อมสร้างกฎเหล็กปลอมว่าห้ามเขียน classification ใน FastAPI และระบุ path `/api/v1/match/internal/scan` ผิด (ของจริงคือ `scan-internal`) ซึ่งขัดกับความเป็นจริงที่ Edge Functions ไม่ได้ถูกเปิดใช้งานในระบบจริง และ Next.js คุยกับ FastAPI โดยตรง
  - `docs/api/api-reference.md` (มี inbound link จาก `docs/INDEX.md`) แสดงตัวอย่าง payload ของ `POST /api/v1/clean` เป็น `{ "texts": [...] }` (array) แต่ Pydantic model ในโค้ดจริง (`models.py:73`) รับ `{ "text": str }` (string เดี่ยว) หากผู้เรียกทำตามเอกสารจะติด HTTP 422 ทันที อีกทั้งเอกสารละเลย router สำคัญที่สุดของระบบคือ `classify.py` และ `internal_match.py`
  - `docs/api/analyze-capabilities.md` อ้างว่าระบบบันทึก feedback ลง `product_deduplication.db` (SQLite) ขัดกับกฎเหล็กข้อ 2 ใน `AGENTS.md` (Supabase is single source of truth, no SQLite dependency)
  - `docs/api/quick-reference.md` ซ้ำซ้อนกับ `api-reference.md` เกือบ 100% ขาด router ปัจจุบัน และไม่มี inbound link จาก `docs/INDEX.md`
- **คำตัดสินที่เสนอ:**
  - `api-reference.md`: เก็บ + แก้ไขใหญ่ (Keep + Rewrite) ให้เป็น Single Source of Truth ของ API docs (แก้ payload ของ clean ให้ถูก, เพิ่ม classify และ internal_match, เพิ่ม Next.js routes)
  - `API_ARCHITECTURE.md`: เขียนใหม่ทั้งหมด (Rewrite) ให้สะท้อนสถาปัตยกรรมจริง (Next.js rewrites -> FastAPI + Supabase DB)
  - `quick-reference.md`: ยุบรวมเข้า `api-reference.md` แล้วลบไฟล์ออก (Consolidate & Delete)
  - `API_CORRECTION.md` และ `analyze-capabilities.md`: ย้ายเข้า `docs/archive_docs/` หรือลบออก (Archive/Delete)
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 10, 2026-10-11):** **แก้/ลบ/จัดเก็บ** — Rewrite `api-reference.md`, Rewrite `API_ARCHITECTURE.md`, Delete `quick-reference.md`, Archive/Delete `API_CORRECTION.md` และ `analyze-capabilities.md` ในตั๋ว 11

## F-029 · P2 · เอกสาร Next.js (`taxonomy-app/app/README.md`, `taxonomy-app/README.md`, `DATABASE_SCHEMA.md`) โฆษณา API routes นำเข้าและรีวิวสินค้าที่ไม่มีอยู่จริง 6 เส้น

- **มิติ:** 1 (เอกสาร vs โค้ด)
- **ชั้น:** เอกสาร (`taxonomy-app/`, `docs/architecture/`)
- **การยืนยัน:** `อ่านโค้ดอย่างเดียว`
- **ไฟล์:บรรทัด:** `taxonomy-app/app/README.md:214-229`, `taxonomy-app/README.md:187`, `docs/architecture/DATABASE_SCHEMA.md:82`, `docs/architecture/REAL_DATABASE_RELATIONSHIPS.md:129`, `taxonomy-app/DEPLOY_LOCAL_MODEL.md:286`
- **ที่มา:**
  - ใน `taxonomy-app/app/README.md` ระบุตาราง Import APIs 5 เส้นที่ไม่มีอยู่จริงใน codebase: `/api/import/process`, `/api/import/process-local`, `/api/import/process-storage`, `/api/import/pending`, `/api/import/approve` (ความจริงขั้นตอนการนำเข้าสินค้าในปัจจุบันถูกยุบรวมเข้า `POST /api/import/commit` ใน `WizardTab.tsx`)
  - ระบุ `POST /api/products/[id]/review` ทั้งใน `app/README.md` และ `taxonomy-app/README.md` ซึ่งไม่มี route นี้อยู่จริง (หน้า `/products` ถูกปรับเป็นแบบอ่านอย่างเดียวตาม `AGENTS.md`)
  - ตารางในเอกสารละเลย route ที่ใช้งานจริงในปัจจุบัน 8 เส้น: `/api/unlock`, `/api/lock`, `/api/verify`, `/api/recheck`, `/api/pos-export`, `/api/settings`, `/api/similarity/resolve`, `/api/keyword-rules/[id]`
- **คำตัดสินที่เสนอ:** แก้ — ปรับปรุงตาราง API endpoints ใน `taxonomy-app/app/README.md` และ `taxonomy-app/README.md` ให้ตรงกับความจริง 25 routes และลบเส้นทางผีในเอกสารสถาปัตยกรรมออกในตั๋ว 11
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 10, 2026-10-11):** **แก้ (Fix)** — ปรับปรุงตาราง API endpoints ใน `taxonomy-app/app/README.md` และ `taxonomy-app/README.md` ให้ตรงกับความจริง 25 routes และลบเส้นทางผีในเอกสารสถาปัตยกรรมออกในตั๋ว 11

## F-030 · P2 · `taxonomy-app/supabase/functions/README.md` โฆษณา Edge Functions ที่ไม่มีอยู่จริง 4 ตัว และแนะนำ OpenAI โมเดลที่ขัดต่อกฎเหล็กโปรเจกต์

- **มิติ:** 1 (เอกสาร vs โค้ด)
- **ชั้น:** เอกสาร (`taxonomy-app/supabase/`)
- **การยืนยัน:** `อ่านโค้ดอย่างเดียว`
- **ไฟล์:บรรทัด:** `taxonomy-app/supabase/functions/README.md:7-66, 86`
- **ที่มา:**
  - เอกสารอธิบายฟังก์ชัน 4 ตัว: `category-suggestions`, `generate-embeddings`, `hybrid-search`, และ `exec-sql` ซึ่งไม่มีอยู่ในโฟลเดอร์จริงแม้แต่ตัวเดียว (ฟังก์ชันที่มีในเครื่องคือ `generate-embeddings-local`, `hybrid-classification-local`, `product-deduplication`)
  - ในตัวอย่างฟังก์ชัน `generate-embeddings` กำหนดให้ส่ง model เป็น `text-embedding-ada-002` พร้อมบังคับใช้ `OPENAI_API_KEY` ขัดกับกฎเหล็กข้อ 3 ใน `AGENTS.md` (384-dim embeddings only, paraphrase-multilingual-MiniLM-L12-v2, no external paid API)
  - `exec-sql` ถูกปลดและลบออกไปแล้วใน migration `20260822000003` เนื่องจากเป็นช่องโหว่ความปลอดภัย แต่เอกสารยังคงสอนให้เรียกใช้งาน
- **คำตัดสินที่เสนอ:** ลบไฟล์นี้ไปพร้อมกับการลบโฟลเดอร์ Edge Functions (ตามข้อเสนอ F-019) ในตั๋ว 10/11
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 10, 2026-10-11):** **ลบ (Delete)** — ลบ `taxonomy-app/supabase/functions/README.md` ทิ้งไปพร้อมกับการลบ Edge Functions ในตั๋ว 11

## F-031 · P3 · สคริปต์และโค้ดตัวอย่างใน `taxonomy-app/` อ้างอิง Edge Functions และ DatabaseService methods ที่ไม่มีอยู่จริง

- **มิติ:** 1 (เอกสาร vs โค้ด) · 6 (เส้นตาย/ซ้ำซ้อน)
- **ชั้น:** โค้ดตัวอย่างและสคริปต์
- **การยืนยัน:** `อ่านโค้ดอย่างเดียว`
- **ไฟล์:บรรทัด:** `taxonomy-app/utils/api.ts:80-155`, `taxonomy-app/test_functions.js:8-88`, `taxonomy-app/scripts/test-api.js:68-128`, `taxonomy-app/data/README_DATASET.md:167,170`
- **ที่มา:**
  - `taxonomy-app/utils/api.ts` มีฟังก์ชันเรียก Edge Functions 6 ตัวที่ไม่มีอยู่จริง (`hybrid-search`, `category-suggestions`, `generate-embeddings`, `similarity-matching`, `batch-processing`, `thai-text-processing`) โดยไฟล์นี้ไม่มีไฟล์อื่นในระบบ import เลย (dead code)
  - `taxonomy-app/test_functions.js` และ `scripts/test-api.js` มีโค้ด curl/fetch ยิงทดสอบ Edge Functions ที่ไม่มีอยู่จริง
  - `taxonomy-app/data/README_DATASET.md` แสดงตัวอย่างการใช้งาน `DatabaseService.getSynonymsByTerm` และ `DatabaseService.getLemmasByCategory` ซึ่งไม่มีเมธอดเหล่านี้อยู่ใน `DatabaseService` จริง
- **คำตัดสินที่เสนอ:** ลบไฟล์ `taxonomy-app/utils/api.ts`, `test_functions.js`, และ `scripts/test-api.js` ในตั๋ว 10/11 และแก้ตัวอย่างโค้ดใน `README_DATASET.md`
- **คำตัดสินที่ผู้ใช้ยืนยัน (ตั๋ว 10, 2026-10-11):** **ลบ (Delete)** — ลบไฟล์ `taxonomy-app/utils/api.ts`, `test_functions.js`, และ `scripts/test-api.js` ในตั๋ว 11 และแก้ตัวอย่างโค้ดใน `README_DATASET.md`

---

## รายการ Type ที่ประกาศซ้ำหลายที่ (สำหรับตั๋ว 08)

| ชื่อ Type / Interface | จำนวนที่พบ | ตำแหน่งไฟล์:บรรทัด |
|---|---|---|
| `TaxonomyNode` | 3 | `app/api/taxonomy/route.ts:11`, `components/data-quality/RecheckTab.tsx:23`, `utils/supabase.ts:25` |
| `Product` | 2 | `components/data-quality/VerifyTab.tsx:24`, `utils/supabase.ts:100` |
| `KeywordRule` | 3 | `app/settings/page.tsx:42`, `components/data-quality/AutoLearnTab.tsx:17`, `utils/supabase.ts:83` |
| `ImportRecord` | 2 | `components/Import/ImportHistory.tsx:20`, `components/Import/StorageImport.tsx:23` |
| `ExportableProduct` | 3 | `app/api/pos-export/route.ts:10`, `utils/pos-export.ts:48`, `__tests__/utils/pos-export.test.ts:12` |
| `PosExportRow` | 3 | `app/api/pos-export/route.ts:11`, `utils/pos-export.ts:58`, `__tests__/utils/pos-export.test.ts:13` |
| `TaxonomyNodeLite` | 3 | `app/api/pos-export/route.ts:12`, `utils/pos-export.ts:42`, `__tests__/utils/pos-export.test.ts:14` |
| `ItemIdCandidate` | 2 | `utils/pos-item-id.ts:42`, `__tests__/utils/pos-item-id.test.ts:11` |
| `ApiResponse` | 2 | `app/api/taxonomy/route.ts:27`, `__tests__/api/simple-api.test.ts:4` |
| `Call` | 3 | `__tests__/api/admin-writes.test.ts:16`, `__tests__/api/pos-export.test.ts:12`, `__tests__/api/verify.test.ts:16` |
| `Filter` | 4 | `__tests__/api/admin-writes.test.ts:15`, `__tests__/api/pos-export.test.ts:11`, `__tests__/api/verify.test.ts:15`, `__tests__/utils/dashboard-stats.test.ts:7` |
| `CategoryRequest` / `CategoryResponse` | 2 | Python `src/api/routers/classify.py:20,43` vs TS `taxonomy-app/types/import.ts` |
| `CleanTextRequest` / `CleanTextResponse` | 2 | Python `src/api/models.py:73,84` vs inline TS `components/Import/DataCleaningStep.tsx:44,49` |


