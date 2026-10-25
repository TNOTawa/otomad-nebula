# Agent Harness Design

Use this when deciding **where a rule should live**.

The default mistake is to encode every concern as natural-language instructions. Prefer the narrowest layer that can actually enforce the concern.

## 1. Constraint placement ladder

### A. Root `AGENTS.md`
Use for a small number of repo-wide, high-value facts and routing pointers.

Good:
- canonical QA command;
- source-of-truth locations;
- project-wide compatibility invariant;
- when to read a deeper document.

Avoid:
- generic software advice;
- long testing philosophy;
- every historical bug;
- rules that only one package needs.

### B. Nested `AGENTS.md`
Use for **subtree deltas**:
- different build/test command;
- local data format;
- package-specific architectural boundary;
- generated directory handling.

Do not copy the root file into every package.

Verify your coding tool’s exact nesting/precedence semantics; support differs.

### C. Skill / on-demand workflow
Use for a repeatable process that only some tasks need:
- grilling / design clarification;
- release;
- migration;
- incident diagnosis;
- code review;
- benchmarking.

The skill description/pointer should clearly state the trigger. Keep its body out of the always-loaded context until invoked.

### D. Focused source-of-truth document
Use for information that must be discoverable but is not an instruction every turn:
- `ARCHITECTURE.md`;
- ADRs;
- data-format specification;
- testing strategy;
- release procedure;
- active execution plan.

The root file should point to it under a concrete condition.

### E. Machine check
If the rule can be evaluated deterministically, prefer:
- test;
- schema;
- type system;
- linter;
- formatter;
- architecture test;
- dependency rule;
- generated consistency check;
- CI gate.

Examples:

```text
“core must not import third-party packages”
    -> architecture/dependency test

“output must be deterministic”
    -> determinism test

“version metadata must agree”
    -> CI consistency check
```

### F. Capability / permission boundary
If violation is dangerous, do not rely only on instructions.

Use:
- sandboxing;
- read-only mounts;
- network restrictions;
- allowlisted tools;
- approval gates;
- secret isolation;
- protected branches;
- CI permissions;
- scoped credentials.

Examples:

```text
“do not deploy production accidentally”
    -> production credential absent + approval gate

“do not rewrite protected test oracle”
    -> agent workspace cannot modify evaluator / protected CI
```

### G. Human decision
Keep genuine product judgment with the human:
- product semantics;
- irreversible trade-offs;
- risk acceptance;
- scope changes;
- policy exceptions.

Use `DECISIONS.md` to avoid asking humans for facts the repository can answer.

## 2. Progressive disclosure

Design agent context like an index:

```text
root AGENTS.md
    -> architecture pointer
    -> decision protocol pointer
    -> quality pointer
    -> local AGENTS.md
    -> task-specific Skill
```

Only the trigger/pointer stays always loaded; branch-specific detail loads when needed.

A useful review question for every root line:

> “Would removing this line change Agent behavior on a meaningful fraction of tasks?”

If not, delete it or move it behind a pointer.

## 3. Positive boundaries over micromanagement

Prefer enforcing invariants while leaving implementation freedom.

Stronger:

```text
UI may depend on Service; Service may depend on Repo; reverse dependencies fail architecture tests.
```

Weaker:

```text
Always write clean layered code and carefully avoid bad dependencies.
```

The first is precise and mechanically checkable.

## 4. No-change is a valid output

The harness should not reward edits merely because a task was opened.

Before changing code, establish enough evidence that a change is required.

Possible successful outcomes include:
- patch produced and verified;
- configuration/docs-only correction;
- issue already fixed / cannot reproduce, with evidence;
- task blocked by a real unresolved decision.

Avoid an unconditional workflow of “receive issue → edit production code”.

## 5. Separate task solving from judging

Where cost/risk justifies it, use separate roles:

```text
Task/spec
   ↓
Implementer
   ↓
Focused tests
   ↓
Independent QA / structural checks
   ↓
High-risk review
```

Do not assume a second LLM is automatically an objective judge. The strongest reviewers have access to independent constraints the implementer cannot redefine.

## 6. Keep the knowledge base alive

Agent-facing docs rot.

Prefer:
- one source of truth;
- cross-link checks;
- generated indexes where appropriate;
- CI checks for required docs/config;
- periodic pruning;
- deletion of rules made redundant by machine enforcement.

When a textual rule becomes fully enforced by code/CI, shorten the instruction to a pointer or remove it.

## 7. A useful hierarchy for new projects

Start small:

```text
AGENTS.md                 # ~map, commands, invariants, pointers
ARCHITECTURE.md           # only when architecture warrants it
docs/adr/                 # consequential durable decisions
docs/agents/
  DECISIONS.md
  QUALITY.md
  HARNESS.md
```

Add nested `AGENTS.md` or more Skills only after a real recurring need appears.

Do not scaffold dozens of empty governance files “just in case”.
