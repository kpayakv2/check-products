# 🔭 Workflow: Socraticode MCP Setup & Quickstart

## 🎯 Objective
เตรียม Socraticode MCP ให้พร้อมใช้งานสำหรับโปรเจกต์ `check-products`
ต้องทำขั้นตอนนี้ **ครั้งแรกของ session** หรือเมื่อ index หายไป

---

## 🔄 Steps

### Step 0: ตรวจสอบสุขภาพ Infrastructure
```
tool: codebase_health
args: {}
```
ตรวจว่า Docker, Qdrant, Ollama, และ Embedding Model พร้อมทำงาน
หากมีปัญหา → แก้ไขตาม error message ก่อนดำเนินการต่อ

---

### Step 1: Index โปรเจกต์ (ครั้งแรก หรือ index เก่า)
```
tool: codebase_index
args:
  projectPath: "d:\\product_checker\\check-products"
```
> ⚠️ `codebase_index` รันใน background — ต้องรอให้ครบ 100% ก่อนใช้ `codebase_search`

ตรวจสอบความคืบหน้าด้วย:
```
tool: codebase_status
args:
  projectPath: "d:\\product_checker\\check-products"
```

---

### Step 2: เปิด Watch Mode (Auto-update index เมื่อแก้ไขไฟล์)
```
tool: codebase_watch
args:
  action: "start"
  projectPath: "d:\\product_checker\\check-products"
```
Watch mode จะ:
- Catch การเปลี่ยนแปลงทุกครั้งที่บันทึกไฟล์
- Update index อัตโนมัติ (debounced)
- ทำ incremental update ก่อน เพื่อ catch changes ที่เกิดขึ้นระหว่าง offline

---

### Step 3: Build Dependency Graph
```
tool: codebase_graph_build
args:
  projectPath: "d:\\product_checker\\check-products"
```
รอให้เสร็จ แล้วตรวจสอบสถิติ:
```
tool: codebase_graph_stats
args:
  projectPath: "d:\\product_checker\\check-products"
```

---

### Step 4: ตรวจสอบ Context Artifacts
```
tool: codebase_context
args:
  projectPath: "d:\\product_checker\\check-products"
```
แสดง artifacts ที่ลงทะเบียนไว้ เช่น `DATABASE_SCHEMA.md`, `API_ARCHITECTURE.md`

---

## 🚀 Quickstart — ค้นหาโค้ดทันที

เมื่อ index เสร็จแล้ว ใช้คำสั่งเหล่านี้ได้เลย:

```
# ค้นหา semantic (ภาษาธรรมชาติ)
codebase_search: "Thai text normalization function"
codebase_search: "hybrid classification keyword embedding"
codebase_search: "pgvector cosine similarity query"

# ดู symbol เฉพาะ
codebase_symbol: { name: "ThaiTextProcessor" }
codebase_symbol: { name: "hybrid_classify" }

# ดู symbols ในไฟล์
codebase_symbols: { file: "fresh_implementations.py" }

# วิเคราะห์ impact ก่อนแก้
codebase_impact: { target: "ThaiTextProcessor" }
codebase_impact: { target: "fresh_implementations.py" }

# Trace execution flow
codebase_flow: { entrypoint: "classify_product" }
```

---

## 📋 Tool Reference สำหรับโปรเจกต์นี้

| Scenario | Tool ที่ใช้ |
|----------|------------|
| ค้นหาโค้ดที่เกี่ยวกับ feature | `codebase_search` |
| ก่อน refactor/ลบ function | `codebase_impact` |
| ดู callers/callees ของ function | `codebase_symbol` |
| List functions ในไฟล์ | `codebase_symbols` |
| Trace flow ของ request | `codebase_flow` |
| ดู circular dependency | `codebase_graph_circular` |
| ดู DB schema / API spec | `codebase_context` |
| เช็ค infra ก่อนเริ่ม | `codebase_health` |

---

## ⚠️ หมายเหตุ Windows

- `projectPath` ต้องใช้ **double backslash** ใน JSON: `"d:\\product_checker\\check-products"`
- รัน Watch mode ทิ้งไว้ตลอด session เพื่อให้ index เป็นปัจจุบันเสมอ
- หาก `codebase_search` ช้ามาก → ตรวจสอบด้วย `codebase_health` ว่า Qdrant ทำงานปกติ
