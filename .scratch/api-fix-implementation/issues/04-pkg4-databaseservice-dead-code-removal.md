# 04: DatabaseService Dead Code Removal

**What to build:**
ชำระล้างโค้ดใน `DatabaseService` (`taxonomy-app/utils/supabase.ts`): ลบ 20 write methods ที่ใช้ anon key และขัดกับนโยบาย RLS (11 เมธอดพังแบบเงียบ เช่น คืน `[]` หรือ 204 และ 9 เมธอดโดนบล็อก 403) และลบ 7 read methods ที่ไม่มีผู้เรียกในหน้าเว็บใดๆ คงไว้เฉพาะ client instance (`supabase`), Type imports กลาง (จาก PKG-1), และเมธอดที่มีผู้เรียกจริงใน frontend

**Blocked by:** 01 (Type SSOT Infrastructure)

**Status:** closed

**ข้อค้นพบที่ปิด (2 ข้อ):**
- #F-025 (P2)
- #F-026 (P3)

**ไฟล์ที่เกี่ยวข้อง:**
- `taxonomy-app/utils/supabase.ts`

**เทสต์ที่ต้องรันก่อน/หลัง:**
- `npx tsc --noEmit` ใน `taxonomy-app/`
- `npx jest --ci` ใน `taxonomy-app/`

- [x] ลบ 20 write methods ใน `DatabaseService`: `updateTaxonomyNode`, `deleteTaxonomyNode`, `updateSynonym`, `deleteSynonym`, `updateImport`, `updateSystemSettings`, `updateRegexRule`, `deleteRegexRule`, `updateKeywordRule`, `deleteKeywordRule`, `getSystemSettings`, `createSimilarityMatch`, `createProduct`, `createProductAttribute`, `createTaxonomyNode`, `createSynonym`, `createSynonymTerm`, `createImport`, `createProductCategorySuggestion`, `createRegexRule`, `createKeywordRule`
- [x] ลบ 7 dead read methods: `getTaxonomyNodes`, `getProducts`, `getSimilarityMatches`, `getTaxonomyNode`, `getNextSortOrder`, `searchTaxonomyNodes`, `getSynonymTerms`
- [x] ปรับให้ `supabase.ts` นำเข้า type จาก `@/types`
- [x] รัน `npx tsc --noEmit` ตรวจสอบว่าไม่มีโค้ดส่วนใดใน UI เรียกเมธอดที่ถูกลบ
- [x] รัน `npx jest --ci` ผ่าน (286 passed / 286 tests)
