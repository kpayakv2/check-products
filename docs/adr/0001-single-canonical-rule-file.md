# Put every project rule in AGENTS.md; CLAUDE.md becomes a one-line import

**Status:** accepted

Three agent tools work on this repo — Claude Code, Antigravity IDE, and (less often) Gemini CLI — and each looks for its own file (`CLAUDE.md`, `AGENTS.md`, `GEMINI.md`). Keeping the actual rule content in `CLAUDE.md` meant Antigravity never saw it, and copying rules into multiple files is exactly what let the fake "72% accuracy" benchmark survive in a dozen documents after it was retracted in one of them (see `docs/CURRENT_STATUS.md`).

We moved the substance of `CLAUDE.md` into `AGENTS.md` — the file Antigravity IDE reads natively (confirmed: Antigravity ≥1.20.5 reads root `AGENTS.md` directly) — and left `CLAUDE.md` as a single `@AGENTS.md` import line, which is Claude Code's documented way to pull in another file's content (Claude Code has no native AGENTS.md support as of mid-2026). `GEMINI.md`'s fate is a separate decision: Gemini CLI can be pointed at `AGENTS.md` too via its own `settings.json`, but that hasn't been done yet.

## Considered options

- **Leave `CLAUDE.md` as the only canonical file.** Rejected — Antigravity IDE, in daily use on this repo, would never see it.
- **Add a fourth, hierarchy-only file and leave `CLAUDE.md`'s content in place.** Rejected — a fifth document doesn't answer "which file is the real one," it just adds another contender.
- **Keep `CLAUDE.md` and `GEMINI.md` independently maintained, synced by hand.** Rejected — this is the status quo that produced the conflicts this decision exists to fix.

## Consequences

- Edit `AGENTS.md`, never `CLAUDE.md` — the latter is a stub, and editing it has no effect on what any tool actually loads.
- Any file that states a rule (docs, `.agents/rules/*`) must link to `AGENTS.md` rather than restate the rule.
- The `@AGENTS.md` import inside `CLAUDE.md` is corroborated by community reports, not confirmed first-hand in this environment yet — verify it in a fresh Claude Code session before treating this as fully closed.
