# =============================================================================
# backup_db.ps1 — Thai Product Taxonomy DB Backup
# =============================================================================
# Backs up runtime tables (products, human_feedback, ml_training_history)
# ที่ Migration SQL ไม่ได้ดูแล → เก็บไว้ใน D:\product_checker\backups\
#
# วิธีใช้:
#   .\scripts\backup_db.ps1                   # Backup ปกติ
#   .\scripts\backup_db.ps1 -Restore          # Restore จาก backup ล่าสุด
#   .\scripts\backup_db.ps1 -List             # ดู backup ที่มีอยู่
# =============================================================================

param(
    [switch]$Restore,
    [switch]$List,
    [string]$RestoreFile = ""
)

# ─── Config ──────────────────────────────────────────────────────────────────
$DB_HOST     = "127.0.0.1"
$DB_PORT     = "54322"          # Supabase Local Postgres direct port
$DB_USER     = "postgres"
$DB_PASS     = "postgres"
$DB_NAME     = "postgres"
$PROJECT_ID  = "product_checker"

# ตาราง runtime ที่ต้อง backup (Migration ดูแล schema, เราดูแล data)
$TABLES = @("products", "human_feedback", "ml_training_history")

# โฟลเดอร์เก็บ backup (อยู่บน D:\ นอก Docker ทั้งหมด)
$BACKUP_DIR = "D:\product_checker\backups"
$KEEP_DAYS  = 7    # เก็บ backup ย้อนหลังกี่วัน (Clean PC Rule)
# ─────────────────────────────────────────────────────────────────────────────

$env:PGPASSWORD = $DB_PASS
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"

# ─── Helper Functions ─────────────────────────────────────────────────────────
function Write-Header {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  Thai Product DB Backup — $TIMESTAMP" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Check-Supabase {
    Write-Host "🔍 ตรวจสอบ Supabase..." -ForegroundColor Yellow
    $result = docker ps --filter "name=supabase_db_$PROJECT_ID" --format "{{.Names}}" 2>&1
    if (-not $result) {
        Write-Host "❌ Supabase ไม่ได้รันอยู่!" -ForegroundColor Red
        Write-Host "   รัน: npx supabase start  ก่อนแล้วลองใหม่" -ForegroundColor Gray
        exit 1
    }
    Write-Host "✅ Supabase พร้อม ($result)" -ForegroundColor Green
}

function Check-PgDump {
    $pgdump = Get-Command pg_dump -ErrorAction SilentlyContinue
    if (-not $pgdump) {
        Write-Host "⚠️  pg_dump ไม่พบบนเครื่อง — ใช้ Docker exec แทน" -ForegroundColor Yellow
        return "docker"
    }
    return "local"
}

function Do-Backup {
    Write-Header
    Check-Supabase

    if (-not (Test-Path $BACKUP_DIR)) {
        New-Item -ItemType Directory -Path $BACKUP_DIR -Force | Out-Null
        Write-Host "📁 สร้างโฟลเดอร์: $BACKUP_DIR" -ForegroundColor Gray
    }

    $pgMode = Check-PgDump
    $successCount = 0

    Write-Host ""
    Write-Host "📦 กำลัง Backup ตาราง Runtime..." -ForegroundColor Yellow
    Write-Host ""

    foreach ($table in $TABLES) {
        $outFile = "$BACKUP_DIR\${TIMESTAMP}_${table}.sql"

        Write-Host "  → $table" -NoNewline

        if ($pgMode -eq "docker") {
            $containerName = "supabase_db_$PROJECT_ID"
            docker exec $containerName pg_dump `
                -U $DB_USER `
                -d $DB_NAME `
                --data-only `
                --table=public.$table `
                2>&1 | Out-File -FilePath $outFile -Encoding utf8
        } else {
            pg_dump `
                -h $DB_HOST `
                -p $DB_PORT `
                -U $DB_USER `
                -d $DB_NAME `
                --data-only `
                --table=public.$table `
                -f $outFile 2>&1
        }

        if (Test-Path $outFile) {
            $size = (Get-Item $outFile).Length
            $sizeKB = [math]::Round($size / 1KB, 1)
            Write-Host " ✅ ($sizeKB KB)" -ForegroundColor Green
            $successCount++
        } else {
            Write-Host " ⚠️  ตารางว่างหรือยังไม่มีข้อมูล" -ForegroundColor Yellow
        }
    }

    Write-Host ""
    Write-Host "📊 สรุป: Backup $successCount/$($TABLES.Count) ตารางสำเร็จ" -ForegroundColor Cyan
    Write-Host "📂 เก็บไว้ที่: $BACKUP_DIR" -ForegroundColor Gray

    # ลบ backup เก่ากว่า KEEP_DAYS วัน (Clean PC Rule)
    $cutoff = (Get-Date).AddDays(-$KEEP_DAYS)
    $oldFiles = Get-ChildItem $BACKUP_DIR -Filter "*.sql" | Where-Object { $_.LastWriteTime -lt $cutoff }
    if ($oldFiles.Count -gt 0) {
        $oldFiles | Remove-Item -Force
        Write-Host "🗑️  ลบ backup เก่ากว่า $KEEP_DAYS วัน: $($oldFiles.Count) ไฟล์" -ForegroundColor Gray
    }

    Write-Host ""
    Write-Host "✅ Backup เสร็จสมบูรณ์!" -ForegroundColor Green
    Write-Host ""
}

function Do-List {
    Write-Host ""
    Write-Host "📂 Backup Files ใน $BACKUP_DIR" -ForegroundColor Cyan
    Write-Host "─────────────────────────────────────────────" -ForegroundColor Gray

    if (-not (Test-Path $BACKUP_DIR)) {
        Write-Host "⚠️  ยังไม่มี backup เลย" -ForegroundColor Yellow
        return
    }

    $files = Get-ChildItem $BACKUP_DIR -Filter "*.sql" | Sort-Object LastWriteTime -Descending
    if ($files.Count -eq 0) {
        Write-Host "⚠️  ไม่พบไฟล์ .sql ใน $BACKUP_DIR" -ForegroundColor Yellow
        return
    }

    $files | ForEach-Object {
        $sizeKB = [math]::Round($_.Length / 1KB, 1)
        $date = $_.LastWriteTime.ToString("yyyy-MM-dd HH:mm")
        Write-Host "  📄 $($_.Name)  ($sizeKB KB)  [$date]" -ForegroundColor White
    }

    Write-Host ""
    Write-Host "รวม: $($files.Count) ไฟล์" -ForegroundColor Gray
    Write-Host ""
}

function Do-Restore {
    Write-Header
    Check-Supabase

    $latest = Get-ChildItem $BACKUP_DIR -Filter "*_products.sql" `
              | Sort-Object LastWriteTime -Descending `
              | Select-Object -First 1

    if (-not $latest) {
        Write-Host "❌ ไม่พบไฟล์ backup ใน $BACKUP_DIR" -ForegroundColor Red
        exit 1
    }

    $dateStamp = $latest.BaseName -replace "_products$", ""
    Write-Host "📅 Restore จาก backup: $dateStamp" -ForegroundColor Yellow
    Write-Host ""

    foreach ($table in $TABLES) {
        $sqlFile = "$BACKUP_DIR\${dateStamp}_${table}.sql"
        if (Test-Path $sqlFile) {
            Write-Host "  → Restore $table ..." -NoNewline
            $containerName = "supabase_db_$PROJECT_ID"
            Get-Content $sqlFile | docker exec -i $containerName psql -U $DB_USER -d $DB_NAME 2>&1 | Out-Null
            Write-Host " ✅" -ForegroundColor Green
        } else {
            Write-Host "  → $table ... ⚠️ ไม่พบไฟล์" -ForegroundColor Yellow
        }
    }

    Write-Host ""
    Write-Host "✅ Restore เสร็จสมบูรณ์!" -ForegroundColor Green
    Write-Host ""
}

# ─── Main ────────────────────────────────────────────────────────────────────
if ($List)         { Do-List }
elseif ($Restore)  { Do-Restore }
else               { Do-Backup }
