# CLAUDE.md — Thai Product Taxonomy Manager & Similarity Checker

Adapted for Claude Code from this repo's original Gemini CLI agent setup (`GEMINI.md`, `.gemini/`, `.agents/`). Those files are kept as reference material — this file is the one Claude Code actually loads.

## Where the project currently stands
Read `docs/CURRENT_STATUS.md` first — it carries the running session log, the measured numbers, and what is in flight. Append a session entry there when you finish a piece of work; it is the handover point between sessions. (Only the legacy `.agents/workflows/` files used to reference it, so it was going unread.)

## Project Overview
AI-powered Thai product taxonomy management + similarity/dedup matching.
- **Backend:** Python + FastAPI (embedding provider, 384-dim)
- **Frontend:** Next.js + Supabase (Edge Functions + pgvector), Tailwind CSS
- **Algorithm:** Hybrid classification — Keyword 60% + Embedding 40%. Measured top-1 sub-category accuracy: **72.3%** on a 595-item held-out test set (2026-08-26), up from 25.5% at the start of that day. Most of the lift comes from `keyword_rules` rows with `match_type='mined_legacy'`, mined from the legacy labelled data by `scripts/mine_keywords_from_legacy.py` — if accuracy collapses, check those rows still exist

## Project Constitution (กฎเหล็ก)
1. **`src/` structure only** — core code lives under `src/`
2. **Supabase is the single source of truth** — no SQLite / `human_feedback.db` dependency
3. **384-dim embeddings only** — `paraphrase-multilingual-MiniLM-L12-v2`, column type `vector(384)`
4. **No Tailwind outside `taxonomy-app/`**
5. **Test before changing code or DB** — run pytest/jest first, never skip
6. **Never regress the measured accuracy baseline** — run `tests/integration/test_classification_accuracy.py` before and after any algorithm change. The old "≥72% F1" rule cited `tests/benchmark_similarity.py`, which printed a hardcoded "72%" without classifying anything; it was deleted 2026-08-26 and replaced by that real test. Raise the baselines in the test when an improvement is proven
7. **`127.0.0.1`, never `localhost`, in Python/FastAPI on Win32** — see Windows rules below

## Known Unresolved Issues (re-verified 2026-08-26)
Original audit: `docs/reports/REPO_AUDIT_2026-08-19.md`. Current status:
1. ~~Edge Function `exec-sql` unauthenticated~~ — **fixed**: function removed, `verify_jwt` lines commented out, migration `20260822000003_drop_exec_sql_function.sql`
2. ~~No `middleware.ts`~~ — **fixed**: `taxonomy-app/middleware.ts` gates every non-GET `/api/*` behind a session cookie
3. ~~`taxonomy-app/jest.config.js` key misspelled `moduleNameMapping`~~ — **fixed**: line 39 now reads `moduleNameMapper`, `@/...` aliases resolve in tests
4. ~~`requirements.txt` lists `sqlite3`~~ — **fixed**

## Legacy Data Workflow (added 2026-08-26)
The 3,103 human-categorised products in `input/รายการสินค้าพร้อมหมวดหมู่_AI.txt` are the system's ground truth. Load them via `src/utils/legacy_dataset.py` — it handles the double encoding (UTF-16 wrapping cp874) and the stratified train/test split. Never mine keywords from the test split.

Pipeline, in order:
1. `scripts/mine_keywords_from_legacy.py` — builds `keyword_rules` (`match_type='mined_legacy'`). `--source all` for production, `--source train` to make accuracy measurable. The accuracy test auto-skips when rules were mined from all data, so re-mine with `--source train` before trusting a number.
2. `scripts/import_legacy_products.py` — loads the 3,103 products with their human categories.
3. `scripts/recheck_legacy_categories.py` — AI re-classifies them, writing `product_category_suggestions` (`suggestion_method='recheck_legacy'`) with `product_id` set. ~20% disagree with the human label.
4. Review the disagreements at **/data-quality → ตรวจซ้ำของเก่า (Recheck)**. Confirming updates `products.category_id` in place, writes `review_history`, and calls `/api/v1/learn/verify` so the system keeps learning.

**Two gotchas that cost real debugging time:**
- FastAPI caches `keyword_rules` at startup. Adding rules to the DB changes nothing until the server reloads (it runs with `reload=True`, so touching a file under `src/` is enough).
- `INTERNAL_API_SECRET` must be set in `taxonomy-app/.env.local` or **every mutating API route returns 401** and the UI silently cannot save. It is not in any committed example file. After setting it, restart `npm run dev` — the Edge middleware reads env at server start, so an API route can see the value while the middleware still does not.

## MCP Tools Available in This Project
Configured in `.mcp.json` (ported from the old `.gemini/settings.json`, minus servers redundant with Claude Code's native tools):
- **postgres** — direct query access to the local Supabase Postgres DB (`127.0.0.1:54325`). Use for real DB inspection instead of guessing schema.
- **socraticode** — codebase impact/symbol/search tools (`codebase_impact`, `codebase_symbol`, `codebase_search`, `codebase_graph_circular`). Use before editing any shared function/module to see blast radius and callers.
- **sequential-thinking** — structured multi-step reasoning for complex logic (e.g. changes to the hybrid classification algorithm).
- **memory** — knowledge-graph memory server (separate from Claude Code's own file-based memory system). Storage pinned via `MEMORY_FILE_PATH` in `.mcp.json` to `.mcp-memory/memory.jsonl` (gitignored) so it survives `npx` cache changes — by default this server writes next to wherever `npx` happens to cache the package, which is not stable across installs.

Not ported: `filesystem` (redundant with Read/Edit/Write/Glob), `puppeteer` (redundant — this repo already has Playwright configured in `taxonomy-app/e2e/`), `domscribe` (unclear purpose, skipped).

## Blast-Radius Rule
Before editing any function, class, or shared module: check callers first. Use the `socraticode` MCP tools if available, otherwise `Grep`/`Glob` or the `Explore` agent for broader searches. Don't assume a change is isolated without checking.

## Windows / PowerShell
- Supabase API Gateway: port `54331` (`54321` is often Windows-reserved)
- Supabase DB port: `54325`
- FastAPI/Python: bind and connect via `127.0.0.1`, not `localhost` (socket binding is unreliable on Win32)
- Frontend `.env.local`: use `http://localhost:3000` for the browser-facing URL (avoids CORS)
- PowerShell command chaining: use `;`, not `&&`

## LAN Access
`NEXT_PUBLIC_*` env vars are bundled into the browser, so `127.0.0.1` in them breaks other LAN machines. Use Next.js rewrites as a reverse proxy and relative paths (`/api/fastapi`, `/api/supabase`) for `NEXT_PUBLIC_*` values; keep absolute `127.0.0.1` URLs only in server-side env vars. Full checklist and firewall commands: `.agents/rules/rules-windows.md`.

## Supabase / Database
- Use the Supabase client with TypeScript generics — no raw SQL in application code, no untyped queries
- RLS must be enabled on every table with an explicit policy
- Vector similarity: cosine distance (`<=>`)
- See `.agents/rules/rules-supabase.md` for query examples

## Thai Text / Product Data
- Normalize all Thai product text through `ThaiTextProcessor` (`fresh_implementations.py`) before comparing — never compare raw strings
- Naming: TypeScript `camelCase`, Python `snake_case`; no `any` in TypeScript
- See `.agents/rules/rules-thai-product.md` for examples

## Git Hygiene
- Never commit files >100MB, `model_cache/`, `node_modules/`, `.next/`, `.env*`, or credential files
- Run `git status` and `git diff --stat` before every commit/push
- Full checklist: `.agents/rules/rules-git-hygiene.md`

## UI Changes
This repo already has Playwright configured (`taxonomy-app/e2e/*.spec.ts`). After any UI change:
- Run the relevant spec, or `npx playwright test` for the full suite
- Check for horizontal scroll at 375px, console errors (`Failed to fetch`), and Thai text rendering (no clipped vowels, long text wrapped/truncated)
- Verify against the real local Supabase instance, not mocks

## Testing
- Python: `.venv/Scripts/python.exe -m pytest` (repo's venv only — system Python lacks pytest). Full suite ~2 min.
- Frontend: `npx jest --ci` in `taxonomy-app/` (~90s); typecheck with `npx tsc --noEmit`
- Expected non-regressions as of 2026-08-29: 6 pytest failures needing a live FastAPI on `:8000`; **4 jest suite failures** (`__tests__/integration/*` and `__tests__/setup/database-setup.ts`) that throw because jest doesn't load `.env.local`; **9 `tsc` errors** confined to `e2e/` and `__tests__/integration/`
- **The Playwright suite is rotted, not a regression signal** — only 2 of 20 tests pass. Most specs assert UI text that no longer exists (e.g. `เลือกวิธีการ Import`), and `e2e/real-user-workflows.spec.ts` fails to collect at all because it imports `__tests__/setup/database-setup.ts`, which throws without env vars. Rewrite it before trusting it as a gate

## Key Directories
| Path | Purpose |
|------|---------|
| `src/api` | Python API (`api_server.py`, `routers/`) |
| `src/core` | Core AI logic (`fresh_implementations.py`, `models.py`) |
| `src/services` | Service layer (`ml_feedback_learning.py`) |
| `scripts` | CLI data-management scripts |
| `taxonomy-app` | Next.js app, UI, Supabase client |
| `docs` | Architecture, API docs, DB schema, reports |
| `supabase` | Edge Functions, migrations |
| `tests` | Pytest unit/integration tests |
| `.agents` | Original Gemini CLI rules/skills/workflows (reference) |

## Reference: Original Agent Config
- `.agents/rules/` — rules-ai-agent, rules-supabase, rules-thai-product, rules-git-hygiene, rules-windows, rules-antigravity
- `.agents/skills/` — thai-taxonomy-expert, data-cleaner, pgvector-semantic-search, vercel-react-best-practices
- `.agents/workflows/` — smart_impact_workflow, workflow-new-feature, workflow-analyze-db, workflow-antigravity-verification
- `.agents/memory/` — logged bugs/lessons (bug_numpy_feature_names, dedup_refactor_lessons)
