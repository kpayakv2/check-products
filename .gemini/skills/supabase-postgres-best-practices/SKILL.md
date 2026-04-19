# Supabase Postgres Best Practices Skill

## Overview
This skill provides procedural knowledge for optimizing Postgres databases within the Supabase ecosystem, specifically focusing on Thai product taxonomy management.

## Key Rules (by Priority)

### 1. Query Performance (CRITICAL) - Prefix: `query-`
- **Missing Indexes:** Always check for missing indexes on frequently filtered columns (e.g., `category_id`, `product_name`).
- **Partial Indexes:** Use partial indexes for large tables where only a subset of data is frequently queried.
- **Explain Analysis:** Use `EXPLAIN ANALYZE` to verify query plans before deployment.

### 2. Security & RLS (CRITICAL) - Prefix: `security-`
- **Row-Level Security (RLS):** Ensure all tables have RLS enabled.
- **Policy Optimization:** Keep RLS policies simple to avoid performance overhead on each row fetch.

### 3. Schema Design (HIGH) - Prefix: `schema-`
- **Data Normalization:** Balance between normalization and performance for high-read product data.
- **Constraints:** Use proper foreign key constraints and check constraints (e.g., for `confidence_score` between 0 and 1).

### 4. Advanced Features (MEDIUM) - Prefix: `advanced-`
- **Full-Text Search:** Leverage Postgres GIN indexes for Thai text searching if embedding search is overkill.

## How to Apply
Invoke this skill when:
- Designing new database schemas for products or taxonomy.
- Optimizing slow queries in the similarity matching engine.
- Writing RLS policies for the Next.js frontend.
