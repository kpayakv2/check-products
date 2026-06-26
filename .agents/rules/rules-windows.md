---
name: rules-windows
description: |
  กฎเฉพาะสำหรับการพัฒนาบน Windows (Win32/PowerShell)
  ครอบคลุม Port Management, Path Handling, LAN Access และ Shell quirks

triggers:
  - รันคำสั่งใน PowerShell
  - ตั้งค่า Port สำหรับ Supabase หรือ FastAPI
  - พบ error ที่เกี่ยวกับ socket, port binding, หรือ CORS
  - เขียน .env หรือ config ที่มี URL/localhost
  - ต้องการให้เครื่องอื่นในวงแลนเข้าใช้งานได้
---

# 🪟 Windows (Win32) Development Rules

## 🔌 Port Management
- **Supabase API Gateway:** ต้องใช้พอร์ต `54331` เสมอ (เนื่องจาก `54321` มักถูก Windows Reserve ไว้)
- **Database Port:** ต้องใช้พอร์ต `54325` เพื่อหลีกเลี่ยงการชนกับ Postgres มาตรฐาน (54322)
- **Frontend URL:** ใน `.env.local` ต้องใช้ `http://localhost:3000` ห้ามใช้ `127.0.0.1` เพื่อป้องกันปัญหา CORS ในเบราว์เซอร์

## 🐍 Python / FastAPI
- ใช้ `127.0.0.1` **แทน** `localhost` เสมอใน Python/FastAPI บน Win32
  ```python
  # ✅ Good
  uvicorn.run(app, host="127.0.0.1", port=8000)
  # ❌ Bad — อาจหา socket ไม่เจอบน Windows
  uvicorn.run(app, host="localhost", port=8000)
  ```

## 🚀 PowerShell Execution
- การรันคำสั่งหลายคำสั่งต่อกันใน PowerShell ให้ใช้ `;` แทน `&&`
  ```powershell
  # ✅ Good
  cd taxonomy-app; npm run dev
  # ❌ Bad — && ไม่ทำงานใน PowerShell เสมอไป
  cd taxonomy-app && npm run dev
  ```
- หากเจอข้อความ `bind: An attempt was made to access a socket...` ให้ตรวจสอบพอร์ตใน `config.toml` ทันที

## 📁 Paths
- ใช้ Path แบบ Windows (Backslash `\`) ในคำสั่ง Shell
- ใช้ Forward Slash `/` ในโค้ด Python/TypeScript เสมอ
- ใช้ `pathlib.Path` ใน Python เพื่อให้ cross-platform

---

## 🌐 LAN Access (บันทึก 2026-05-30)

### หลักการสำคัญ
ตัวแปร `NEXT_PUBLIC_*` ถูก bundle ลงใน **browser** → เครื่อง LAN อื่นอ่านค่า `127.0.0.1` แล้วพยายามเชื่อมหาตัวเอง ❌

**วิธีแก้มาตรฐาน:** ใช้ Next.js เป็น Reverse Proxy + ตัวแปรเป็น relative path

### Checklist ทุกครั้งที่ต้องการ LAN Access

**1. Next.js — ต้อง bind 0.0.0.0**
```batch
npx next dev --hostname 0.0.0.0
```

**2. next.config.js — เพิ่ม rewrites proxy**
```js
async rewrites() {
  const fastapiUrl = process.env.FASTAPI_URL || 'http://127.0.0.1:8000'
  const supabaseUrl = process.env.SUPABASE_URL || 'http://127.0.0.1:54331'
  return [
    { source: '/api/fastapi/:path*',  destination: `${fastapiUrl}/:path*`  },
    { source: '/api/supabase/:path*', destination: `${supabaseUrl}/:path*` },
  ]
}
```

**3. .env.local — NEXT_PUBLIC_* ต้องเป็น relative path**
```env
# ❌ ห้ามใช้ 127.0.0.1 ใน NEXT_PUBLIC_
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000

# ✅ ใช้ relative path แทน
NEXT_PUBLIC_API_URL=/api/fastapi
NEXT_PUBLIC_SUPABASE_URL=/api/supabase

# Server-side ยังใช้ 127.0.0.1 ได้ปกติ (ไม่ bundle ลง browser)
FASTAPI_URL=http://127.0.0.1:8000
SUPABASE_URL=http://127.0.0.1:54331
```

**4. Supabase Client — ต้องใช้ absolute URL (SDK ไม่รับ relative path)**
```ts
// utils/supabase.ts
const getSupabaseUrl = (): string => {
  if (typeof window === 'undefined') {
    return process.env.SUPABASE_URL || 'http://127.0.0.1:54331'  // SSR
  }
  return `${window.location.origin}/api/supabase`  // Browser → LAN-aware อัตโนมัติ
}
export const supabase = createClient(getSupabaseUrl(), supabaseAnonKey)
```

**5. Firewall — เปิด port ที่จำเป็น (ต้อง Run as Admin)**
```batch
netsh advfirewall firewall add rule name="Phayak-Frontend-3000"  protocol=TCP dir=in localport=3000  action=allow
netsh advfirewall firewall add rule name="Phayak-Backend-8000"   protocol=TCP dir=in localport=8000  action=allow
netsh advfirewall firewall add rule name="Phayak-Supabase-54331" protocol=TCP dir=in localport=54331 action=allow
```

**6. LAN IP ที่ถูกต้อง — ใช้ route print แทน ipconfig**
```batch
# ✅ ได้ IP จริงของ default gateway interface (ไม่ติด Hyper-V/WSL)
for /f "tokens=4" %%a in ('route print 0.0.0.0 ^| findstr " 0.0.0.0 "') do set LAN_IP=%%a

# ❌ ipconfig อาจได้ virtual adapter (172.x.x.x) ก่อน physical NIC
```

### LAN IP ของเครื่อง PHAYAK Server
- **IP จริง:** `192.168.1.80`
- **UI Dashboard:** `http://192.168.1.80:3000`
- **AI Backend:** `http://192.168.1.80:8000/docs`

### Architecture Flow
```
เครื่อง LAN (browser)
  └─▶ http://192.168.1.80:3000            [Next.js]
        ├─▶ /api/fastapi/*  → :8000       [FastAPI AI Engine]
        └─▶ /api/supabase/* → :54331      [Supabase PostgreSQL]
```

> ⚠️ ทุกครั้งที่แก้ `next.config.js` ต้อง **restart Next.js** ก่อน config จะมีผล
> ⚠️ `START_PHAYAK.bat` ต้องรันแบบ **Run as Administrator** เพื่อให้ Firewall rules ทำงานได้
