# Thai Product Taxonomy Manager & Similarity Checker

AI-powered Thai product taxonomy management and similarity/dedup matching. This glossary currently covers how the project's own documentation is organized — the vocabulary below lets any session, or any of the agent tools used on this repo, decide which file wins when two documents disagree.

## Language

**Rule**:
A behavioral constraint — something an agent must or must not do. Lives in `AGENTS.md`, or in a file `AGENTS.md` explicitly names as where the detail for one rule lives. Never duplicated — every other mention of a rule links back instead of restating it.
_Avoid_: constitution, กฎเหล็ก (fine as a spoken concept, just don't use it to name a second home for a rule)

**Status**:
A point-in-time measurement or progress log — what's currently true, not what must be true. Can go stale without being wrong (it was accurate when written), but must never be read as a Rule.
_Avoid_: benchmark, target (when the words are used to describe a past measurement rather than a requirement)

**Wayfinding**:
A document whose only job is pointing a reader — human or agent — to where to start or where something lives. Carries no Rule or Status content of its own, only links and one-line descriptions.
_Avoid_: onboarding doc (once it starts asserting requirements or numbers, it has stopped being this)

**Memory**:
A fallible, per-session or per-user observation about the project's past state. Never authoritative over a Rule or Status document, regardless of what it claims — verify against current reality before acting on it. Includes Claude Code's auto-memory (`MEMORY.md` and its linked files) and `.agents/memory/`.
_Avoid_: source of truth, canonical (memory is never either)
