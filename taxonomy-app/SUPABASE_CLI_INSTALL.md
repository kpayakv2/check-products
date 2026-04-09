# 🚀 คู่มือติดตั้ง Supabase CLI

## 📋 วิธีติดตั้ง Supabase CLI

### 🔧 วิธีที่ 1: ใช้ npm (แนะนำ)
```bash
npm install -g supabase
```

### 🔧 วิธีที่ 2: ใช้ Chocolatey (Windows)
```bash
# ติดตั้ง Chocolatey ก่อน (ถ้ายังไม่มี)
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# ติดตั้ง Supabase CLI
choco install supabase
```

### 🔧 วิธีที่ 3: ใช้ Scoop (Windows)
```bash
# ติดตั้ง Scoop ก่อน (ถ้ายังไม่มี)
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
irm get.scoop.sh | iex

# ติดตั้ง Supabase CLI
scoop bucket add supabase https://github.com/supabase/scoop-bucket.git
scoop install supabase
```

### 🔧 วิธีที่ 4: Download Binary โดยตรง
1. ไปที่ [Supabase CLI Releases](https://github.com/supabase/cli/releases)
2. Download ไฟล์ `.exe` สำหรับ Windows
3. วางไฟล์ใน PATH หรือโฟลเดอร์ที่ต้องการ
4. เพิ่ม PATH ใน Environment Variables

### 🔧 วิธีที่ 5: ใช้ PowerShell Script
```powershell
# Download และติดตั้งอัตโนมัติ
$url = "https://github.com/supabase/cli/releases/latest/download/supabase_windows_amd64.zip"
$output = "$env:TEMP\supabase.zip"
$extractPath = "$env:LOCALAPPDATA\supabase"

# Download
Invoke-WebRequest -Uri $url -OutFile $output

# Extract
Expand-Archive -Path $output -DestinationPath $extractPath -Force

# Add to PATH (ชั่วคราว)
$env:PATH += ";$extractPath"

# หรือเพิ่ม PATH ถาวร
[Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";$extractPath", [EnvironmentVariableTarget]::User)
```

## ✅ ตรวจสอบการติดตั้ง

หลังจากติดตั้งแล้ว ให้ทดสอบด้วยคำสั่ง:

```bash
supabase --version
```

ควรแสดงผลลัพธ์คล้ายนี้:
```
supabase version 1.123.4
```

## 🔐 การ Login และ Setup

### 1. Login เข้า Supabase
```bash
supabase login
```

### 2. Link กับ Project
```bash
# ใน taxonomy-app directory
supabase link --project-ref YOUR_PROJECT_REF
```

### 3. ตรวจสอบการเชื่อมต่อ
```bash
supabase status
```

## 📊 คำสั่งที่จำเป็นสำหรับ Taxonomy App

### 🗄️ Database Commands
```bash
# รัน migrations
supabase db reset

# Push schema ไปยัง remote
supabase db push

# Pull schema จาก remote
supabase db pull

# Generate types
supabase gen types typescript --local > types/supabase.ts
```

### ⚡ Edge Functions Commands
```bash
# Deploy ทุก functions
supabase functions deploy

# Deploy function เดียว
supabase functions deploy hybrid-search
supabase functions deploy category-suggestions
supabase functions deploy generate-embeddings

# ดู logs
supabase functions logs hybrid-search
```

### 🔑 Secrets Management
```bash
# ตั้งค่า secrets สำหรับ Edge Functions
supabase secrets set OPENAI_API_KEY=your_openai_key
supabase secrets set HUGGINGFACE_API_KEY=your_hf_key

# ดู secrets ที่มี
supabase secrets list
```

### 🏃‍♂️ Local Development
```bash
# เริ่ม local development
supabase start

# หยุด local development
supabase stop

# ดูสถานะ
supabase status
```

## 🛠️ Setup สำหรับ Thai Product Taxonomy Manager

### 1. เตรียม Project
```bash
# ใน taxonomy-app directory
cd d:\product_checker\check-products\taxonomy-app

# Link กับ Supabase project
supabase link --project-ref YOUR_PROJECT_REF
```

### 2. Setup Database
```bash
# รัน schema และ dataset
supabase db reset
# หรือ
supabase db push
```

### 3. Deploy Edge Functions
```bash
# Deploy AI functions
supabase functions deploy hybrid-search
supabase functions deploy category-suggestions  
supabase functions deploy generate-embeddings

# ตั้งค่า API keys
supabase secrets set OPENAI_API_KEY=your_key
```

### 4. Generate Types
```bash
# สร้าง TypeScript types
supabase gen types typescript --local > types/supabase.ts
```

## 🚨 Troubleshooting

### ❌ ปัญหา: Command not found
**แก้ไข**: ตรวจสอบ PATH environment variable

### ❌ ปัญหา: Permission denied
**แก้ไข**: รันใน Administrator mode หรือใช้ `sudo` (Linux/Mac)

### ❌ ปัญหา: Network error
**แก้ไข**: ตรวจสอบ internet connection และ firewall

### ❌ ปัญหา: Login failed
**แก้ไข**: ตรวจสอบ credentials และลอง login ใหม่

## 📚 คำสั่งที่มีประโยชน์

```bash
# ดูความช่วยเหลือ
supabase help

# ดู project info
supabase projects list

# ดู functions ที่ deploy แล้ว
supabase functions list

# ดู database migrations
supabase migration list

# Backup database
supabase db dump --file backup.sql

# Restore database
supabase db reset --file backup.sql
```

## 🎯 Next Steps

หลังจากติดตั้ง Supabase CLI แล้ว:

1. ✅ Login เข้า Supabase account
2. ✅ Link กับ taxonomy-app project  
3. ✅ Deploy Edge Functions
4. ✅ Setup environment variables
5. ✅ Test การเชื่อมต่อ

---

## 📞 การสนับสนุน

หากมีปัญหาในการติดตั้ง:
- ดู [Supabase CLI Documentation](https://supabase.com/docs/guides/cli)
- ตรวจสอบ [GitHub Issues](https://github.com/supabase/cli/issues)
- ใช้ `supabase help` สำหรับความช่วยเหลือ
