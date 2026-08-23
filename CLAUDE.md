# CLAUDE.md — Thai Product Taxonomy Manager & Similarity Checker

Adapted for Claude Code from this repo's original Gemini CLI agent setup (`GEMINI.md`, `.gemini/`, `.agents/`). Those files are kept as reference material — this file is the one Claude Code actually loads.

## Project Overview
AI-powered Thai product taxonomy management + similarity/dedup matching.
- **Backend:** Python + FastAPI (embedding provider, 384-dim)
- **Frontend:** Next.js + Supabase (Edge Functions + pgvector), Tailwind CSS
- **Algorithm:** Hybrid classification — Keyword 60% + Embedding 40% → target accuracy ≥ 72%

## Project Constitution (กฎเหล็ก)
1. **`src/` structure only** — core code lives under `src/`
2. **Supabase is the single source of truth** — no SQLite / `human_feedback.db` dependency
3. **384-dim embeddings only** — `paraphrase-multilingual-MiniLM-L12-v2`, column type `vector(384)`
4. **No Tailwind outside `taxonomy-app/`**
5. **Test before changing code or DB** — run pytest/jest first, never skip
6. **Similarity benchmark ≥ 72% F1** — algorithm changes must not regress this
7. **`127.0.0.1`, never `localhost`, in Python/FastAPI on Win32** — see Windows rules below

## Known Unresolved Issues (verify before trusting — status as of 2026-08-19)
See `docs/reports/REPO_AUDIT_2026-08-19.md` for full detail. As of the last check, none of these were fixed:
1. Edge Function `exec-sql` (`taxonomy-app/supabase/config.toml`) has `verify_jwt = false` + raw `EXECUTE query_text` + service_role — unauthenticated arbitrary SQL if deployed
2. `taxonomy-app/app/api/` has no `middleware.ts` — no central auth layer on API routes
3. `taxonomy-app/jest.config.js` — key is misspelled `moduleNameMapping` (should be `moduleNameMapper`); path aliases (`@/...`) aren't actually mapped in tests
4. `requirements.txt` lists `sqlite3`, which isn't on PyPI — breaks `pip install` in CI

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
- Expected non-regressions as of 2026-08-19 (see `docs/reports/REPO_AUDIT_2026-08-19.md`): 6 pytest failures needing a live FastAPI on `:8000`, 8 jest suite failures needing live Supabase + the `jest.config.js` typo above, 11 `tsc` errors confined to `e2e/` test code

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
