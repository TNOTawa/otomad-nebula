# Decision Protocol

Use this only when a task contains **material unresolved decisions**, or when the user explicitly asks for a grilling / stress-test session.

The goal is not to maximize questions. The goal is to minimize unresolved intent before expensive or irreversible work begins.

## 1. Run an ambiguity gate first

Classify each uncertainty as one of four kinds:

### FACT
Something that can be discovered from the repository, tools, logs, documentation, APIs, or a small experiment.

**Action:** investigate it. Do not ask the user to act as a search engine.

Examples:
- Which serializer is already used?
- What does this SDK parameter mean?
- Which test command covers this package?
- Does a similar implementation already exist?

### PRECEDENT
A choice already implied by a project convention, stable architecture, accepted ADR, or nearby implementation.

**Action:** follow the precedent unless the task explicitly changes it. Mention the precedent if it materially shapes the solution.

### REVERSIBLE LOCAL CHOICE
A low-consequence choice that is cheap to change later and has a clear project-default answer.

**Action:** choose the least-surprising existing convention and proceed. Do not create ceremonial questions.

### CONSEQUENTIAL DECISION
A choice that materially changes product behavior, architecture, compatibility, data, security, scope, or irreversible cost, and is not resolved by FACT/PRECEDENT.

**Action:** ask the user.

## 2. Normal mode: targeted clarification

For ordinary coding work, do not “grill” by default.

Ask only the smallest set of questions required to resolve consequential ambiguity. Prefer a question that distinguishes between concrete behaviors over a broad “what do you want?” question.

Good:

```text
The existing API treats an omitted `limit` as “unbounded”.
Should the new endpoint preserve that behavior, or introduce the proposed default of 100?
I recommend preserving “unbounded” for compatibility and making 100 an explicit opt-in.
```

Weak:

```text
Can you clarify the requirements?
```

If no material ambiguity remains, proceed.

## 3. Grill mode: stress-test decisions before implementation

Enter this mode when the user explicitly invokes `/grill-me`, asks to be grilled, or asks for a thorough design pressure-test.

If a compatible grilling skill is installed, use it rather than duplicating its full prompt here.

Operational pattern:

1. Map the decision tree.
2. Resolve upstream decisions before dependent decisions.
3. Look up FACTS yourself.
4. Put actual DECISIONS to the user.
5. For each question, give a recommended answer and the main trade-off.
6. In a live interactive session, prefer one decision at a time so later questions can depend on earlier answers.
7. Do not begin the implementation phase until the user says the design has enough shared understanding.

The last two points are **interaction heuristics**, not universal requirements for every task.

## 4. Convert decisions into durable constraints

Once a decision is settled, place it at the narrowest useful layer:

- One-off task choice → task/spec/plan.
- Long-lived architectural rationale → ADR/design doc.
- Subtree-specific operational rule → nested `AGENTS.md`.
- Reusable workflow → Skill.
- Behavior that can be checked → test/property/contract.
- Architecture invariant → architecture test/linter.
- Dangerous action → permission/sandbox/tool policy.
- Global project fact → root map or a pointed-to source of truth.

Do not keep a decision as repeated chat folklore if it can be made discoverable or enforceable.

## 5. Detect contradictory or underspecified oracles

Tests are evidence, not automatically the specification.

If:
- the task says A,
- existing tests require B,
- and A and B cannot both be true,

treat this as a specification conflict. Do not “win” by weakening the test or silently ignoring the task.

Follow the repository’s documented authority rule. If none exists and the difference is consequential, ask the user.

## 6. Decision log for complex work

For long-running or multi-agent changes, keep a small checked-in plan or ADR when useful:

```text
Decision:
Why this matters:
Options considered:
Chosen:
Consequences:
How verified:
```

Record decisions that future work will need; do not archive every conversational detail.
