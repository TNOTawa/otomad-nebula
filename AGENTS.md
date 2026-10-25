# AGENTS.md

> **This file is a map, not a manual.**
> Keep it short, stable, project-specific, and mechanically actionable.
> Put branch-specific knowledge behind pointers; put enforceable rules into tests, CI, linters, architecture checks, or tool permissions.

## Sources of truth

- The current task/spec describes intended behavior.
- Existing code and tests describe current behavior and compatibility obligations.
- Project docs describe architecture, data semantics, and rationale.
- Local/nested `AGENTS.md` files may add subtree-specific facts and commands.
- If these sources conflict, do not silently pick the easiest one. Follow an explicitly documented authority rule; otherwise surface the conflict as a decision.

## Default work protocol

1. **Investigate** — inspect the relevant implementation, tests, docs, and existing call sites before assuming behavior.
2. **Verify need** — establish whether a code change is actually required. A verified **no-change** result is a valid successful outcome.
3. **Resolve uncertainty** — look up facts yourself. Ask only about consequential decisions that remain unresolved after investigation.
4. **Change narrowly** — make the smallest coherent change; reuse existing abstractions and single sources of truth.
5. **Verify independently** — run focused checks while developing, then the project QA gate before completion.
6. **Report evidence** — state what changed, what was run, what passed, what could not be run, and any remaining risk.

## Engineering invariants

- Preserve user data, backward compatibility, and stable identities unless the task explicitly changes them.
- Confirm unfamiliar APIs, units, encodings, indexes, lifetimes, and boundary semantics from code/docs/tests or a minimal experiment.
- Keep dependency direction and architecture boundaries intact.
- Prefer deterministic behavior for generators, compilers, serializers, exporters, and reports when the domain allows it.
- Treat tests, golden files, quality thresholds, and CI rules as part of the verification system. Changes to them require task intent and explicit justification.
- Prefer rules that can be mechanically checked over rules that merely ask the Agent to “be careful”.
- An inability to run a required check is a reported limitation, not an implicit pass.

## Read on demand

- **Material product/architecture ambiguity remains, or the user asks to stress-test a plan / invokes a grilling workflow**  
  → read `docs/agents/DECISIONS.md`.
- **Adding/changing tests, QA, coverage, mutation testing, golden files, or deciding whether work is complete**  
  → read `docs/agents/QUALITY.md`.
- **Changing CI, permissions, sandboxes, architecture checks, agent infrastructure, or instruction layout**  
  → read `docs/agents/HARNESS.md`.
- **Need the evidence behind these rules**  
  → read `docs/agents/RESEARCH.md`.
- **Working inside a subtree with its own `AGENTS.md`**  
  → read that file and treat it as the local delta; do not duplicate root guidance into it.

## Project map

> Fill these in for each repository. Delete fields that do not apply.

- **Project:** TODO
- **Purpose:** TODO
- **Main entry point:** TODO
- **Architecture map:** TODO (prefer a pointer, e.g. `ARCHITECTURE.md`)
- **Build/setup:** TODO
- **Fast relevant tests:** TODO
- **Full QA command:** TODO — prefer one canonical entry point such as `python qa.py`, `make qa`, or `npm run qa`
- **Release/build artifact:** TODO
- **Critical data/compatibility notes:** TODO (prefer pointers to focused docs)

## Completion

Completion is based on **observable evidence**, not Agent confidence.

A normal completion report should be compact and factual, for example:

```text
Changed: ...
Focused checks: ...
Full QA: ...
Golden/determinism: ...
Not run / remaining risk: ...
```
