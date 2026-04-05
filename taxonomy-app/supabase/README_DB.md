# Database Migrations (Hybrid UUID + Code)

Authoritative schema now uses UUID primary keys with separate stable `code` columns for human readable references.

## Active Migration Chain
1. `20250924140000_init_hybrid_schema.sql` – Core tables, indexes, triggers.
2. `20250924140010_load_taxonomy.sql` – Inserts taxonomy nodes (root + subset of sub-categories; extend as needed).
3. `20250924140020_load_synonyms.sql` – Inserts synonym lemmas & terms via code→UUID lookups.
4. `20250924140030_load_rules.sql` – Inserts keyword & regex rules + system settings.

## Deprecated
Earlier text-primary-key migrations were moved to `migrations/_archive` (or deleted) on 2025-09-24. Retrieve from git history if required.

## Conventions
- PK: `UUID` (internal, never exposed as external contract unless necessary).
- Code patterns:
  - Categories: `cat_###[_###]` (e.g. cat_003_007)
  - Lemmas: `lemma_###`
  - Keyword Rules: `rule_kw_###`
  - Regex Rules: `rule_rx_###`
- Add new records by assigning a new code, letting UUID auto-generate.

## Adding a New Sub-Category
```
INSERT INTO taxonomy_nodes (code, name_th, name_en, level, parent_id, sort_order, keywords)
VALUES ('cat_003_015','ตัวอย่าง','Example',1,(SELECT id FROM taxonomy_nodes WHERE code='cat_003'),15, ARRAY['ตัวอย่าง','example']);
```

## Adding a Lemma + Terms
```
INSERT INTO synonym_lemmas (code, name_th, name_en, description, category_id, is_verified)
VALUES ('lemma_099','ตัวอย่าง','Example','คำอธิบาย',(SELECT id FROM taxonomy_nodes WHERE code='cat_003_001'), true);

INSERT INTO synonym_terms (lemma_id, term, language, is_primary, confidence_score)
VALUES ((SELECT id FROM synonym_lemmas WHERE code='lemma_099'),'คำตัวอย่าง','th', true, 0.95);
```

## Rules Example
```
INSERT INTO keyword_rules (code, name, description, keywords, category_id, match_type, priority, confidence_score)
VALUES ('rule_kw_099','Example Rule','ตัวอย่าง', ARRAY['คำ','ตัวอย่าง'], (SELECT id FROM taxonomy_nodes WHERE code='cat_003_001'),'contains',5,0.80);
```

## Dev Reset (Supabase CLI)
Use when safe to drop all data:
```
supabase db reset
```

## Verification Queries
```
SELECT COUNT(*) taxonomy FROM taxonomy_nodes;
SELECT COUNT(*) lemmas FROM synonym_lemmas;
SELECT COUNT(*) terms FROM synonym_terms;
SELECT COUNT(*) kw_rules FROM keyword_rules;
SELECT COUNT(*) rx_rules FROM regex_rules;
SELECT COUNT(*) settings FROM system_settings;
```

## Next Improvements
- Add RLS policies & roles (ported selectively from old snapshot) as a new migration.
- Add audit trail (audit_logs + triggers) in a separate migration for clarity.
- Expand remaining sub-categories & lemmas by importing full dataset.

## Rationale
Hybrid model protects internal referential integrity (UUID) while preserving stable business-facing codes. Renames or structural reorganizations now avoid PK churn.
