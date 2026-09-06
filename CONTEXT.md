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
A fallible observation about the project's past state — never authoritative over a Rule or Status document, regardless of what it claims. Verify against current reality before acting on it. Splits into Personal Memory and Team Memory below.
_Avoid_: source of truth, canonical (memory is never either)

**Personal Memory**:
Claude Code's own auto-memory (`MEMORY.md` + linked files), stored outside the repo, visible only to you, injected into every session. The most-read Memory, so a stale claim here does the most damage — fix it the moment you find it wrong, in the same turn.
_Avoid_: shared memory, team notes

**Team Memory**:
Hand-written lessons in `.agents/memory/`, checked into the repo. Visible to anyone who opens the repo, in any agent tool — not auto-injected, has to be read on purpose.
_Avoid_: personal notes, session log
