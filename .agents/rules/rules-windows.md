# 🪟 Windows (Win32) Development Rules

## 🔌 Port Management
- **Supabase API Gateway:** ต้องใช้พอร์ต `54331` เสมอ (เนื่องจาก `54321` มักถูก Windows Reserve ไว้)
- **Database Port:** ต้องใช้พอร์ต `54325` เพื่อหลีกเลี่ยงการชนกับ Postgres มาตรฐาน (54322)
- **Frontend URL:** ใน `.env.local` ต้องใช้ `http://localhost:3000` ห้ามใช้ `127.0.0.1` เพื่อป้องกันปัญหา CORS ในเบราว์เซอร์

## 🚀 Execution
- การรันคำสั่งหลายคำสั่งต่อกันใน PowerShell ให้ใช้ `;` แทน `&&`
- หากเจอข้อความ `bind: An attempt was made to access a socket...` ให้ตรวจสอบพอร์ตใน `config.toml` ทันที

## 📁 Paths
- ใช้ Path แบบ Windows (Backslash `\`) ในคำสั่ง Shell แต่ใช้ Forward Slash `/` ในโค้ดเสมอ
