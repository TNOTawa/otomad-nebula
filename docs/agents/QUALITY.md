# Quality and Verification

The principle is:

> **Trust independent evidence more than the implementation author’s confidence.**

A test suite is one oracle, not the whole truth. Build a verification **portfolio** whose components fail for different reasons.

## 1. One QA entry point

Prefer one canonical command that orchestrates project-relevant checks:

```text
python qa.py
make qa
npm run qa
```

The root `AGENTS.md` should point to it instead of repeating every subcommand.

During development, run the narrowest relevant checks first. Before completion, run the canonical QA command when feasible.

## 2. Verification portfolio

Choose layers according to risk; not every project needs every layer.

### Functional
- unit tests
- integration tests
- acceptance / Gherkin tests
- regression tests
- negative/error-path tests

### Semantic / invariant
- property-based tests
- round-trip tests
- determinism checks
- schema/contracts
- stable-ID / monotonicity / state invariants

### Artifact / compatibility
- golden tests
- compatibility fixtures
- migration tests
- package/install/smoke tests

### Structural
- architecture/dependency tests
- type checking
- static analysis
- lint
- generated-file consistency checks

### Test-strength signals
- changed-line / branch coverage
- mutation testing
- flaky-test tracking

Coverage and mutation score are **signals**, not definitions of correctness.

## 3. Prefer independent oracles

Where practical, separate implementation from evaluation.

Stronger patterns include:
- human-authored reproduction/acceptance tests for important behavior;
- hidden or protected CI checks for high-value invariants;
- an evaluator/audit that checks artifact structure as well as behavior;
- architecture checks the implementation agent cannot casually bypass;
- independent fixtures/golden data derived from a trusted source.

The reason is simple: an agent that sees the scoring signal can optimize the signal itself.

## 4. Protect the verifier

Do not rely only on a prompt saying “do not cheat”.

Prefer repository or CI mechanisms that make the following changes visible, gated, or impossible without explicit intent:

- deleting/weakening tests;
- lowering coverage or mutation thresholds;
- broad `skip`/`ignore`;
- re-recording golden files;
- disabling static/security checks;
- altering evaluator configuration.

A legitimate test/gate change is allowed when the task genuinely changes behavior or verification policy, but it should be explicit and justified.

## 5. Mutation testing: use selectively

Mutation testing is valuable where a small logic error matters and ordinary coverage can look healthy despite weak assertions.

Prioritize:
- parsers;
- selection/ranking logic;
- boundary arithmetic;
- serializers/deserializers;
- state transitions;
- core business rules.

For a surviving mutant:
1. decide whether it is truly equivalent;
2. if not, identify the missing observable behavior;
3. strengthen the test;
4. rerun.

Equivalent-mutant exceptions should carry a reason.

Do not force mutation testing onto UI glue or slow integration paths if its cost outweighs its signal.

## 6. Golden tests: protect meaning, not snapshots

A golden diff is a question, not permission to re-record.

When a golden changes:
1. explain the intended semantic change;
2. verify it with a more local assertion when possible;
3. inspect the diff;
4. only then update the golden.

## 7. Multi-oracle completion

For important tasks, avoid a single “all tests green” stopping rule.

A stronger finish can combine:

- task/acceptance behavior;
- structural audit;
- compatibility/golden result;
- property/invariant;
- full QA;
- targeted human review for high-risk boundaries.

This reduces both “test hacking” and “building to the test”.

## 8. Risk-based human review

Automated constraints should free human attention, not eliminate judgment.

Increase human review for:
- destructive migrations;
- auth/security/cryptography;
- concurrency/lifetimes;
- external protocols;
- financial/irreversible state;
- major architecture changes;
- weakly tested legacy core.

Low-risk, well-constrained changes can rely much more heavily on the harness.

## 9. Evidence-based completion report

Report observed outcomes:

```text
Focused tests: 42 passed
Full QA: passed
Mutation: 91%, gate 85%
Golden: unchanged
Architecture check: passed
Not run: Windows packaging (environment unavailable)
```

Do not replace missing evidence with “should be fine”.
