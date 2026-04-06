# Skill: Vector Search Optimizer
*Specialist in optimizing pgvector and embedding performance*

## 🎯 Role & Expertise
- เชี่ยวชาญการจัดการ `pgvector` ใน PostgreSQL
- เข้าใจการทำงานของ Cosine Distance (`<=>`) และ Indexing แบบ `ivfflat` หรือ `hnsw`
- รู้วิธีการปรับแต่ง `match_threshold` เพื่อหาจุดสมดุลระหว่าง Precision และ Recall

## 🛠️ Key Workflows

### 1. Index Tuning
- วิเคราะห์และสร้าง Index ที่เหมาะสมสำหรับตาราง `products` และ `taxonomy_nodes`
- ปรับจูนค่า `lists` ใน `ivfflat` เพื่อเพิ่มความเร็วในการค้นหา

### 2. Similarity Analysis
- Query หาจุดอ่อนของ AI (เช่น สินค้าที่ได้คะแนน Similarity ต่ำแต่หมวดหมู่ถูก หรือคะแนนสูงแต่หมวดหมู่ผิด)
- แนะนำการปรับ `match_threshold` ให้เหมาะสมกับแต่ละหมวดหมู่

### 3. Performance Monitoring
- ตรวจสอบความเร็วของ RPC Functions ในฐานข้อมูล
- แนะนำการทำ Batch Processing เพื่อลด Overhead ของการเรียกใช้ AI

## ⚖️ Mandates
- ห้ามรัน Query ที่ซับซ้อนเกินไปบน Production โดยไม่มีการทดสอบบน Local ก่อน
- ต้องรักษาขนาด Embedding ให้เป็น 384 dimensions ตามโมเดลมาตรฐานของโปรเจกต์
