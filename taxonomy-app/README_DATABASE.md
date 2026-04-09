# Database Schema & Migrations - Thai Product Taxonomy

## Schema Overview

This project uses a **Hybrid UUID+Code** approach for database design:

- **Primary Keys**: UUID (internal, immutable, system-generated)
- **Human Codes**: TEXT (external references, human-readable, stable)
- **Benefits**: Combines UUID flexibility with human-readable references

## Schema Architecture

### Core Tables

1. **taxonomy_nodes** - Product categories hierarchy
   - `id` UUID PK (internal)
   - `code` TEXT UNIQUE (cat_001, cat_001_001, etc.)
   - Supports parent-child relationships via UUID foreign keys

2. **synonym_lemmas** - Synonym groups 
   - `id` UUID PK (internal)
   - `code` TEXT UNIQUE (lemma_001, lemma_002, etc.)
   - Links to taxonomy via UUID foreign key

3. **synonym_terms** - Individual terms in synonym groups
   - `id` UUID PK (internal)
   - `lemma_id` UUID FK → synonym_lemmas.id

4. **keyword_rules** & **regex_rules** - Classification rules
   - `id` UUID PK (internal) 
   - `code` TEXT UNIQUE (rule_kw_001, rule_rx_001, etc.)
   - Links to taxonomy via UUID foreign key

5. **system_settings** - System configuration (UUID only)

### Supporting Tables
- **products** - Product records (UUID PK, category_id → taxonomy_nodes.id)
- **product_category_suggestions** - ML/rule-based suggestions
- **imports**, **audit_logs**, **product_attributes**, **review_history**, **similarity_matches**

## Migration Strategy

### Current Migration Chain

1. **20250925003812_init_hybrid_schema.sql** - Complete schema with UUID PKs + code columns
2. **20250924130000_load_taxonomy.sql** - Load taxonomy hierarchy using code lookups
3. **20250924130010_load_synonyms.sql** - Load synonym lemmas & terms using code lookups  
4. **20250924130020_load_rules.sql** - Load keyword/regex rules + system settings using code lookups

### Data Loading Pattern

All dataset loaders use the pattern:
```sql
-- Insert with code, lookup UUID for foreign keys
INSERT INTO child_table (code, name, parent_id) VALUES 
('child_code', 'Child Name', (SELECT id FROM parent_table WHERE code = 'parent_code'));
```

### Archived Migrations

Legacy migrations moved to `_archive/` folder:
- `20250924114400_create_schema.sql` - Old TEXT PK approach
- `20250924114425_install_datasets.sql` - Old monolithic dataset loader
- `20250924123255_load_full_datasets.sql` - Old combined loader

## Reset & Deployment

### Local Reset
```powershell
cd taxonomy-app
supabase db reset
```

### Verify Data Load
```sql
-- Check record counts
SELECT 
  'taxonomy_nodes' as table_name, COUNT(*) as count
FROM taxonomy_nodes
UNION ALL
SELECT 
  'synonym_lemmas' as table_name, COUNT(*) as count  
FROM synonym_lemmas
UNION ALL
SELECT 
  'synonym_terms' as table_name, COUNT(*) as count
FROM synonym_terms
UNION ALL
SELECT 
  'keyword_rules' as table_name, COUNT(*) as count
FROM keyword_rules
UNION ALL
SELECT 
  'regex_rules' as table_name, COUNT(*) as count
FROM regex_rules
UNION ALL
SELECT 
  'system_settings' as table_name, COUNT(*) as count
FROM system_settings
ORDER BY table_name;
```

Expected counts after full load:
- taxonomy_nodes: ~53 (12 main + 41+ sub-categories)
- synonym_lemmas: ~15 
- synonym_terms: ~40+
- keyword_rules: ~25
- regex_rules: ~10
- system_settings: 1

## Code Conventions

### ID Patterns
- **Taxonomy**: `cat_001` (main), `cat_001_001` (sub-category)
- **Synonym Lemmas**: `lemma_001`, `lemma_002`, etc.
- **Keyword Rules**: `rule_kw_001`, `rule_kw_002`, etc. 
- **Regex Rules**: `rule_rx_001`, `rule_rx_002`, etc.

### Adding New Categories

1. Add to taxonomy loader with new code:
```sql
INSERT INTO taxonomy_nodes (code, name_th, name_en, level, parent_id, ...) 
VALUES ('cat_013', 'New Category', 'New Category EN', 0, NULL, ...);
```

2. Add synonym lemmas if needed:
```sql  
INSERT INTO synonym_lemmas (code, name_th, category_id, ...)
VALUES ('lemma_016', 'New Synonyms', (SELECT id FROM taxonomy_nodes WHERE code = 'cat_013'), ...);
```

3. Add classification rules:
```sql
INSERT INTO keyword_rules (code, name, category_id, ...)
VALUES ('rule_kw_026', 'New Detection', (SELECT id FROM taxonomy_nodes WHERE code = 'cat_013'), ...);
```

## Security & Permissions

- **RLS Enabled**: All tables have Row Level Security policies
- **Roles**: `taxonomy_reader`, `taxonomy_editor`, `taxonomy_admin`
- **Audit Logging**: Automatic logging of all changes via triggers
- **API Exposure**: Use `code` fields for external references, keep UUIDs internal

## Development Guidelines

1. **Never expose UUID PKs** in APIs - use `code` fields
2. **Always use code lookups** in dataset loaders for foreign keys  
3. **Maintain code uniqueness** across environments
4. **Test with reset** before deployment
5. **Document code patterns** when adding new entity types

## Troubleshooting

### Migration Errors
- Check foreign key references use proper lookups: `(SELECT id FROM table WHERE code = 'xxx')`
- Ensure parent records loaded before children in migration order
- Verify no duplicate codes within same entity type

### Performance
- All `code` columns have UNIQUE indexes
- Use code-based queries for external APIs
- Use UUID-based queries for internal joins

## Schema Source

Original schema based on `schema.sql` with these key modifications:
1. Added `code TEXT UNIQUE NOT NULL` columns to domain tables
2. Renamed `synonyms` → `synonym_lemmas` for dataset compatibility  
3. Updated all foreign key references to use UUID lookups
4. Maintained all original indexes, triggers, RLS policies, and roles
