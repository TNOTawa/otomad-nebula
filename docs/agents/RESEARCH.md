# Research Rationale

This file explains **why** the starter kit is structured this way. It is for maintainers and should not be always-loaded into every Agent turn.

Research status matters:
- **Peer-reviewed** findings get the most weight.
- **Workshop / early empirical** work is useful but narrower.
- **Preprints** are treated as signals, not settled facts.
- **Industry practice / Skills** provide implementation patterns, not scientific proof.

## Evidence matrix

| Design choice | Evidence | Status | What we adopt |
|---|---|---|---|
| Keep repository instruction files minimal | Gloaguen et al., *Evaluating AGENTS.md* found no task-success improvement overall and >20% inference-cost increase; unnecessary requirements made tasks harder | ICLR 2026 MemAgents workshop, Oral / Runner-up Best Paper | Root `AGENTS.md` is a map, not an encyclopedia |
| Useful AGENTS context can reduce navigation cost | Lulla et al. report lower median runtime and output tokens with AGENTS.md while task completion was comparable | 2026 preprint / ICSE JAWs presentation | Keep exact commands and high-value project facts; do not infer “AGENTS is always bad” |
| Long multi-constraint instructions are difficult | FollowBench and ComplexBench identify weaknesses as fine-grained constraints accumulate/compose | ACL 2024; NeurIPS 2024 | Reduce always-loaded constraint count; route branch-specific rules |
| Progressive disclosure / map-not-manual works in agent-first practice | OpenAI’s Codex harness experiment reports that one giant AGENTS.md failed; they use ~100-line map + structured docs + mechanical checks | Industry evidence, OpenAI 2026 | Root pointers + focused docs + mechanical enforcement |
| Ask clarifying questions only when ambiguity is real | ClarifyGPT detects ambiguous requirements first and asks targeted questions; asking on unambiguous tasks adds burden and may hurt solutions | PACMSE/FSE 2024 | Add an ambiguity gate; do not run `/grill-me` for every task |
| Interactive intent formalization can improve code generation | TiCoder uses test-driven user feedback and reports lower cognitive load in a user study and large pass@1 gains in automated evaluation | IEEE TSE 2024 | Turn important decisions into acceptance examples/tests when useful |
| `/grill-me`: facts from environment, decisions from user | Matt Pocock’s current `grilling` Skill explicitly separates discoverable facts from decisions and resolves decision dependencies | Practitioner heuristic | Adopt FACT/PRECEDENT/DECISION split; keep grilling on-demand |
| `/grill-me`: one question at a time | Matt Pocock Skill | Practitioner heuristic; no direct strong SE experiment found in this review | Use in explicit interactive grill mode, not as universal coding rule |
| Narrow, test-driven agent workflows can outperform monolithic agents | TDFlow decomposes patch proposing/debugging/revision under constrained tools and human-written tests; reports 88.8% SWE-Bench Lite and 94.3% Verified test-resolution pass rates | EACL 2026 long paper | Prefer focused workflows and independent tests over giant prompts |
| Visible tests can be gamed | ImpossibleBench measures agents exploiting tests, including test modification and more subtle shortcuts | ICLR 2026 | Protect evaluators/gates; tests are not the sole oracle |
| Even hidden behavioral tests may not prove the requested artifact exists | *Building to the Test* reached near-perfect hidden oracle scores while the requested reusable library could remain dead/absent | Microsoft 2026 preprint; narrow study | Add structural/semantic audits; use multiple independent oracles |
| Mutation testing can strengthen real test suites | Google-scale study of ~15M mutants found developers improved tests and mutants were coupled with real faults | ICSE 2021 | Use mutation testing selectively on high-value logic |
| Prompt-only safety cannot guarantee tool safety | *Towards Verifiably Safe Tool Use for LLM Agents* argues model safeguards cannot guarantee safety and proposes enforceable tool/data-flow constraints | ICSE-NIER 2026 | Put dangerous-action constraints in permissions/sandboxes/tool policies |
| Coding agents have action bias | FixedBench reports unnecessary edits on 35–65% of already-fixed tasks; “reproduce first” helps but can over-abstain on partially fixed tasks | 2026 research / COLM workshop track | Explicitly allow verified no-change; do not blindly force reproduction or edits |

## What we deliberately did **not** turn into root rules

### “Grill relentlessly before every task”
Rejected as a default.

ClarifyGPT’s central design choice is deciding **when** to ask. Its paper explicitly notes that questioning both ambiguous and unambiguous requirements creates unnecessary interaction and may result in incorrect code solutions.

Therefore:
- normal mode = targeted clarification only;
- explicit `/grill-me` mode = deep decision interview.

### “All tests passing means correct”
Rejected.

ImpossibleBench demonstrates specification gaming against tests. *Building to the Test* further shows that even a hidden behavioral oracle can be satisfied while the requested artifact structure is wrong.

Therefore quality uses a **portfolio**: behavior + invariants + structure + compatibility + protected evaluation where risk warrants it.

### “Mutation score / coverage is the truth”
Rejected.

They are useful signals, not semantic correctness proofs. A project should set thresholds according to risk and cost, and pair them with other oracles.

### “Never review Agent code”
Not encoded as doctrine.

A strong harness can reduce line-by-line review for low-risk work, but human judgment remains valuable for destructive, security-sensitive, concurrent, protocol, financial, and major architectural changes.

### “Always reproduce before modifying”
Not encoded as an unconditional rule.

FixedBench shows action bias, but also reports a complementary failure mode where strong reproduce-before-patch prompting can cause over-abstention on partially fixed issues. The root rule is narrower: **verify whether a change is actually required; no-change is allowed.**

## Similar mechanisms worth learning

### 1. Nested instruction files
The AGENTS.md convention supports project/subproject scoping; Claude Code similarly supports hierarchical/lazy project memory. This is useful for local deltas, but exact precedence/load behavior differs by tool. Verify your tool before relying on it.

### 2. Agent Skills
Skills move repeatable workflows out of always-loaded instructions and load them when invoked. Matt Pocock’s `writing-for-agents` explicitly frames this as balancing context load and human cognitive load. This is consistent with progressive disclosure, though the exact Skill design rules are practitioner engineering rather than controlled research.

### 3. ADRs / checked-in execution plans
Durable decisions and complex plans can live in versioned repository artifacts rather than chat history. OpenAI’s harness-engineering report uses design docs and execution plans as first-class repo knowledge. This is strong industry practice, not a causal academic result by itself.

### 4. Architecture tests / custom linters
If a dependency direction or data-boundary rule is critical, encode it as a machine check. OpenAI reports using structural tests and custom linters for layer/dependency invariants.

### 5. Capability boundaries
Use sandbox/network/tool/credential permissions for dangerous operations. This has stronger safety semantics than natural-language “never do X” rules.

### 6. Independent evaluators
For important agent work, separate implementation and evaluation when possible. The evaluator should own constraints the implementer cannot freely redefine.

## Primary references

1. Gloaguen, T. et al. (2026). **Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents?** ICLR 2026 Workshop on Memory for LLM-Based Agentic Systems.  
   https://arxiv.org/abs/2602.11988

2. Lulla, J. L. et al. (2026). **On the Impact of AGENTS.md Files on the Efficiency of AI Coding Agents.**  
   https://arxiv.org/abs/2601.20404

3. Jiang, Y. et al. (2024). **FollowBench: A Multi-level Fine-grained Constraints Following Benchmark for Large Language Models.** ACL 2024.  
   https://aclanthology.org/2024.acl-long.257/

4. Wen, B. et al. (2024). **Benchmarking Complex Instruction-Following with Multiple Constraints Composition (ComplexBench).** NeurIPS 2024.  
   https://arxiv.org/abs/2407.03978

5. Mu, F. et al. (2024). **ClarifyGPT: A Framework for Enhancing LLM-Based Code Generation via Requirements Clarification.** Proceedings of the ACM on Software Engineering / FSE.  
   https://doi.org/10.1145/3660810

6. Fakhoury, S. et al. (2024). **LLM-Based Test-Driven Interactive Code Generation: User Study and Empirical Evaluation.** IEEE Transactions on Software Engineering.  
   https://www.microsoft.com/en-us/research/?p=1094508

7. Han, K. et al. (2026). **TDFlow: Agentic Workflows for Test Driven Development.** EACL 2026.  
   https://aclanthology.org/2026.eacl-long.70/

8. Zhong, Z., Raghunathan, A., Carlini, N. (2026). **ImpossibleBench: Measuring LLMs' Propensity of Exploiting Test Cases.** ICLR 2026.  
   https://proceedings.iclr.cc/paper_files/paper/2026/hash/ca688eb14e29701a11bdba6633186328-Abstract-Conference.html

9. Ma, Y., Kereopa-Yorke, B., Schultz, B. (2026). **Building to the Test: Coding Agents Deliver What You Check, Not What You Requested.** arXiv preprint.  
   https://arxiv.org/abs/2606.28430

10. Petrovic, G. et al. (2021). **Long Term Effects of Mutation Testing.** ICSE 2021.  
    https://research.google/pubs/long-term-effects-of-mutation-testing/

11. Doshi, A. et al. (2026). **Towards Verifiably Safe Tool Use for LLM Agents.** ICSE-NIER 2026.  
    https://doi.org/10.1145/3786582.3786839

12. Gloaguen, T. et al. (2026). **Coding Agents Don't Know When to Act.**  
    https://arxiv.org/abs/2605.07769

13. OpenAI (2026). **Harness engineering: leveraging Codex in an agent-first world.**  
    https://openai.com/index/harness-engineering/

14. Matt Pocock Skills. **grilling** and **writing-for-agents**.  
    https://github.com/mattpocock/skills/blob/main/skills/productivity/grilling/SKILL.md  
    https://github.com/mattpocock/skills/blob/main/skills/productivity/writing-for-agents/SKILL.md

15. AGENTS.md open convention.  
    https://agents.md/
