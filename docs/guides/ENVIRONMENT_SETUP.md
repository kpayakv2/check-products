# 🛡️ Clean PC Developer Guide (Environment & Troubleshooting)
*(แนวทางการดูแลสภาพแวดล้อมระบบและการบำรุงรักษาคอมพิวเตอร์สำหรับนักพัฒนา)*

เอกสารฉบับนี้จัดทำขึ้นเพื่อช่วยแก้ปัญหา "เครื่องรก" หรือ "ระบบรวน" ในการพัฒนาโปรเจกต์ บนระบบปฏิบัติการ Windows (Win32) เพื่อรักษาความเป็นระเบียบและลดการสะสมไฟล์ขยะในระบบคอมพิวเตอร์อย่างมีวินัย

---

## 1. 🛡️ การป้องกันการลง Library ผิดพลาด (Virtualenv Guard)

**ปัญหา:** เผลอรัน `pip install` โดยลืม Activate Virtual Environment ทำให้อุปกรณ์หรือโมดูลต่างๆ ไปติดตั้งอยู่บน Windows Global Python ส่งผลให้เครื่องรกและเกิด Library ตีกัน

### **💡 วิธีป้องกันอัตโนมัติ (Force PIP to require Venv):**
คุณสามารถบังคับให้ `pip` ปฏิเสธการทำงานหากไม่พบ Virtual Environment ด้วยการตั้งค่าค่าตัวแปรระบบ (Environment Variable) ดังนี้:

#### **สำหรับ PowerShell (ในเซสชันปัจจุบัน):**
```powershell
$env:PIP_REQUIRE_VIRTUALENV = "true"
```

#### **สำหรับ Windows System-wide (แนะนำ):**
1. เปิดเมนู Start ค้นหา **"Edit the system environment variables"**
2. คลิกปุ่ม **Environment Variables...**
3. ภายใต้หัวข้อ **User variables** คลิก **New...**
4. ตั้งค่าดังนี้:
   * **Variable name:** `PIP_REQUIRE_VIRTUALENV`
   * **Variable value:** `true`
5. คลิก **OK** และเปิด Terminal ใหม่ทั้งหมด

> [!TIP]
> เมื่อเปิดใช้งานตัวแปรนี้แล้ว หากคุณลืมพิมพ์ `.\.venv\Scripts\activate` แล้วไปสั่ง `pip install` ระบบจะตอบกลับทันทีว่า:  
> `Could not find an activated virtualenv (required).`  
> ป้องกันข้อผิดพลาดจากมนุษย์ได้ 100%! หากต้องการลง Global จริงๆ ค่อยระบุ `--isolated` ในคำสั่ง pip

---

## 2. 🐋 ภัยเงียบเรื่องพื้นที่ฮาร์ดดิสก์ (WSL2 / Docker Disk Bloat)

**ปัญหา:** ระบบ Docker บน Windows รันผ่าน WSL2 ซึ่งมีไฟล์ดิสก์เสมือนชื่อ `ext4.vhdx` ดิสก์นี้จะขยายขนาดขึ้นเมื่อมีข้อมูลเพิ่ม (เช่นดึงโมดูล AI หรือเขียน DB) แต่เมื่อสั่ง `docker system prune` หรือลบคอนเทนเนอร์ทิ้ง **ขนาดไฟล์ `.vhdx` จะไม่ยอมลดลงอัตโนมัติ** ทำให้ฮาร์ดดิสก์เต็มโดยไม่รู้ตัว

### **💡 วิธีการทวงคืนพื้นที่ดิสก์เสมือนบน Windows (Shrink vhdx):**
หากพบว่าพื้นที่ดิสก์บนไดรฟ์ C: หรือ D: หายไปเยอะ ให้ทำตามขั้นตอนนี้เพื่อบีบอัดคืนพื้นที่:

1. **ปิดการทำงานของ WSL ทั้งหมด:**
   ```powershell
   wsl --shutdown
   ```
2. **เปิดระบบจัดการดิสก์ของ Windows (Diskpart):**
   * เปิด PowerShell ด้วยสิทธิ์ **Run as Administrator**
   * พิมพ์คำสั่ง:
     ```powershell
     diskpart
     ```
3. **เลือกไฟล์ดิสก์เสมือนของ WSL2:**
   *(โดยทั่วไปจะอยู่ที่ `%USERPROFILE%\AppData\Local\Docker\wsl\data\ext4.vhdx`)*
   * รันคำสั่งระบุที่ตั้งไฟล์ (ปรับเปลี่ยนตามชื่อ User หรือไดรฟ์จริง):
     ```diskpart
     select vdisk file="C:\Users\<ชื่อผู้ใช้งานของคุณ>\AppData\Local\Docker\wsl\data\ext4.vhdx"
     ```
4. **ทำสัญญาบีบอัดดิสก์ (Compact):**
   ```diskpart
   compact vdisk
   ```
5. **เสร็จสิ้นกระบวนการ:**
   ```diskpart
   detach vdisk
   exit
   ```
เพียงเท่านี้คุณจะได้พื้นที่ฮาร์ดดิสก์คืนมาหลาย GB ทันทีหลังจากลบคอนเทนเนอร์ขยะออกไป

---

## 3. 🧟 วิธีล้างโปรเซสซอมบี้ที่ค้างพอร์ต (Zombie Process Hunter)

**ปัญหา:** บางครั้งเมื่อเราปิดหน้าต่าง Terminal หรือกดยกเลิกสคริปต์ไปแล้ว แต่ Node.js (Next.js) หรือ Python (FastAPI) ยังคงรันเบื้องหลังแบบเงียบๆ ทำให้วันต่อมารันระบบไม่ขึ้นเนื่องจากแจ้งเตือนพอร์ตชน (`Port already in use`)

### **💡 วิธีค้นหาและปิดโปรเซสบน Windows (PowerShell):**

#### **ขั้นตอนที่ 1: ค้นหา PID (Process ID) ของพอร์ตที่โดนยึดครอง**
ใช้คำสั่งนี้เพื่อหาว่าใครยึดพอร์ต `3000` (Next.js), `8000` (FastAPI), หรือ `54331` (Supabase Local API):
```powershell
# ค้นหาพอร์ต 3000
Get-NetTCPConnection -LocalPort 3000 | Select-Object LocalAddress, LocalPort, OwningProcess, State

# หรือค้นหาพอร์ต 8000
Get-NetTCPConnection -LocalPort 8000 | Select-Object LocalAddress, LocalPort, OwningProcess, State
```
*มองหาตัวเลขในแถว `OwningProcess` (เช่น `15204` นั่นคือ PID ของโปรเซสที่ค้างคาอยู่)*

#### **ขั้นตอนที่ 2: สั่งบังคับปิดโปรเซสนั้นๆ (Kill Process)**
เมื่อได้เลข PID แล้ว สามารถสั่งทำลายได้ทันที:
```powershell
# ตัวอย่าง PID 15204
Stop-Process -Id 15204 -Force
```

#### **ทางลัด: สั่งหยุดงานในบรรทัดเดียว (One-liner script)**
```powershell
# บังคับล้างโปรเซสที่ใช้พอร์ต 8000 ทันที
Stop-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess -Force
```

---

## 4. 🗄️ การตรวจสอบการผูกข้อมูลฐานข้อมูล (Docker Volume Guard)

**ปัญหา:** เพื่อลบ Container ขยะทิ้งได้ตลอดเวลาโดยที่ข้อมูลสินค้าที่เราตรวจสอบ/จำแนกไปแล้วไม่หายไปกับตา เราต้องมั่นใจว่า Supabase ผูกข้อมูลออกไปเก็บที่ Host (เครื่องเราจริง) ไม่ใช่เก็บไว้ข้างใน Docker Container (NFR3)

### **💡 วิธีการตรวจสอบ:**
1. ตรวจสอบไฟล์ [config.toml](file:///d:/product_checker/check-products/taxonomy-app/supabase/config.toml)
2. มั่นใจว่าได้แมป Volume ไปยังไดเรกทอรีโลคอลจริง
3. ก่อนการสั่งสั่ง `supabase stop` หรือ `docker-compose down` ทุกครั้ง หากต้องการล้างขยะแต่เก็บ Data ไว้ ให้ตรวจสอบว่าไม่รันด้วยคำสั่งที่มีแฟล็ก `-v` (Volume Delete) ยกเว้นในยามที่ต้องการ Reset ฐานข้อมูลเพื่อเริ่มต้นใหม่ทั้งหมดจริงๆ

---

## 📝 ตารางสรุปวินัยที่พึงปฏิบัติประจำวัน (Clean Developer Daily Habits)

| ช่วงเวลา | สิ่งที่ต้องทำ | ประโยชน์ที่ได้รับ |
| :--- | :--- | :--- |
| **ก่อนเริ่มทำงาน** | ตรวจสอบว่า `Docker Desktop` เปิดอยู่ และรัน `START_PHAYAK.bat` | ป้องกันการลืมเปิด Service ต่างๆ และเลี่ยงปัญหา Port ชนสะสม |
| **ก่อนลงแพ็กเกจ** | เช็คว่า Terminal แสดงผล `(.venv)` เสมอ | ป้องกันไฟล์ขยะไหลเข้าไปปะปนใน Windows System |
| **หลังเสร็จงาน** | สั่งหยุด Docker และรันสคริปต์ล้างขยะเป็นระยะ | รักษาสภาพดิสก์เสมือน (WSL2) ไม่ให้บวมตัว |
| **ก่อน Commit** | รัน Pytest ตรวจสอบการทำความสะอาดชื่อภาษาไทย | มั่นใจว่าโค้ดที่แก้นั้นปลอดภัยและพร้อมใช้งานบนโปรดักชัน |
