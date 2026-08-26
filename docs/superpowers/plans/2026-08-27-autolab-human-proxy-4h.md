# AutoLab Co-Evolve + Human Proxy 4h Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch all 17 AutoLab CPU tasks as a Co-Evolve + context-only Human Proxy arm, with ten concurrent 4-hour runs and seven wave-scheduled pending runs.

**Architecture:** Generate one immutable, auditable private Human Proxy context per AutoLab task from the already accepted Final Evaluator package (`final_verifier.py`, runtime context template, authored design, and known issues). Pre-seed each new run with the corrected-control copy of AutoLab's originally shipped weak V0, checker and seed, then resume without `--freeze-verifier` so hardening remains enabled. Extend the generic batch launcher to map each run to its own out-of-tree context path plus SHA-256 and fail closed on drift across initial launch, pending-wave promotion, and crash resume.

**Tech Stack:** Python 3.11, pytest, Co-Scientist `AgentSystem`, Docker-backed Codex agents, JSON batch manifests.

---

## Chunk 1: Per-run Human Proxy context plumbing

### Task 1: Persist and resume a distinct immutable context per launcher run

**Files:**
- Modify: `coscientist/coevo/launcher.py`
- Test: `tests/test_coevo.py`

- [ ] Add failing tests creating two input directories and two different context files, then assert each dry-run/argv contains only its matching `--human-proxy-context` path and the manifest records its SHA-256.
- [ ] Add failing tests for missing/drifted context on initial spawn, pending-to-running promotion and remaining-budget respawn, plus an idempotent `start` whose requested context mapping differs from the persisted contract.
- [ ] Run the focused test and confirm it fails because `LaunchSpec`/`Batch.start` cannot carry a per-run context.
- [ ] Add `human_proxy_context` and `human_proxy_context_sha256` to `LaunchSpec`, persist both per run in `batch.json`, restore them in `_spec_of`, verify before every `_spawn`, and append the matching CLI flag in `_build_argv`.
- [ ] Add first-class launcher flags `--human-proxy-context-dir`, `--human-agent-timeout-s`, `--human-agent-model`, and `--human-agent-reasoning-effort`; resolve `<context-dir>/<input-basename>.md`, require regular non-symlinked files outside all inputs and `runs/`, and pass the exact per-run mapping into `Batch.start`.
- [ ] Add first-class Solver identity flags (or an equivalent validated environment contract) and assert the resolved Solver gateway is exactly model `gpt-5.6-sol` with reasoning effort `high`; persist these resolved values in `batch.json` so an idempotent start cannot silently change them.
- [ ] Reject an existing batch when the newly requested inputs, hours, CPU pool, Human Proxy context mapping/hashes, or first-class Human Agent configuration differs from `batch.json`.
- [ ] Verify focused launcher tests and existing wave/resume tests pass.

## Chunk 2: AutoLab true-evaluator context artifacts

### Task 2: Generate and audit all 17 private evaluator contexts

**Files:**
- Create: `coscientist/experiments/autolab_human_proxy.py`
- Create: `tests/test_autolab_human_proxy.py`
- Modify: `coscientist/coevo/human_proxy_sessions.py`
- Test: `tests/test_model_backed_human_proxy.py`

- [ ] Add failing tests for deterministic context rendering, all Final Evaluator artifact boundaries, SHA-256 inventory, 200,000-character limit, exact old10/new7 task coverage, accepted-report binding, restrictive permissions, and refusal to reuse an incomplete/mismatched output directory.
- [ ] Add a failing Human Proxy test proving a long verbatim span from private evaluator context cannot be returned in `message` or structured outcome fields, while high-level semantic advice remains allowed.
- [ ] Run the tests and confirm the generator is missing.
- [ ] Implement the fixed accepted old10/new7 inventory and `prepare`/`audit` commands. Context names are `autolab_<task>.md`; content is bounded read-only text with file boundaries for `final_verifier.py`, `final_ctx.template.json`, `authored_design.md`, and `issues.md` from `autolab_final_verifier_review_20260825/{old10|new7}/<task>/`.
- [ ] Bind generation to the accepted exact 17-task report, write the context directory with mode 0700 and files with mode 0600, and write `manifest.json` containing arm, task, context SHA-256, every source artifact SHA-256, byte/character counts, accepted report SHA-256, and package path.
- [ ] Add a normalized long-span leak guard at the Human Proxy output boundary before any response/outcome can enter the durable transcript or `HUMAN_GUIDANCE.md`.
- [ ] Verify generator tests and audit a temporary 17-task fixture.

### Task 3: Pre-seed evolvable runs with shipped weak V0

**Files:**
- Modify: `coscientist/experiments/autolab_human_proxy.py`
- Test: `tests/test_autolab_human_proxy.py`

- [ ] Add failing tests that prepare a run from the corrected-control source and assert byte-identical V0/checker/seed, no `bootstrap_start`, shipped-V0 origin, `freeze_verifier=false`, a non-frozen Co-Evolve solver brief, and a no-proof reframe policy.
- [ ] Implement `prepare-runs` to create all 17 fresh run directories from `runs/autolab_shippedv0_control17_4h_v2_20260825__autolab_<task>` without copying candidates/results/outcomes.
- [ ] Add an `audit-runs` command that binds each prepared V0 SHA/checker test SHA/seed SHA to corrected Control, proves `can_resume`, proves no bootstrap event exists, proves hardening is enabled, and proves no private context content/path exists below the run directory.
- [ ] Verify prepare/audit tests and refuse to overwrite any existing run.

### Task 3b: Replay this arm with the accepted Final Evaluators

**Files:**
- Modify: `coscientist/experiments/autolab_human_proxy.py`
- Test: `tests/test_autolab_human_proxy.py`

- [ ] Add failing tests for an `evaluate-results` command that accepts this batch, extracts each run's final best payload, rejects missing/ambiguous tasks, and binds each replay to the same accepted evaluator package, runtime context template, checker and task image.
- [ ] Implement the 17-task replay/report path without the frozen-control-only assumptions of `replay_shippedv0_control17_final.py`; emit per-task result, metric direction/meaning, package hashes and aggregate completeness.
- [ ] Verify the replay command can be invoked before completion as a contract/dry-run and can score completed fixture runs.

## Chunk 3: Release and launch

### Task 4: Verify, deploy, and start the 4h batch

**Files:**
- Modify only through commits: the files above
- Runtime artifacts on A100: `_human_proxy_experiments/autolab_hp_4h_20260827/` and `runs/autolab_hp_4h_20260827/`

- [ ] Run `pytest -q tests` on A100 and require zero failures.
- [ ] Commit and push the implementation branch, then fast-forward/cherry-pick it into the A100 experiment worktree without disturbing untracked historical artifacts.
- [ ] Generate the 17 contexts from `autolab_final_verifier_review_20260825/{old10|new7}/<task>/`, verify the accepted report/package hashes, pre-seed 17 fresh evolvable run dirs, and pass both audits.
- [ ] Pin comparator-parity run settings: Solver model `gpt-5.6-sol`, Solver reasoning effort `high`, `--solver-strength weak`, `--feedback with_artifacts`, existing per-task resource/timeouts, and 4h; pin Human Agent settings to `gpt-5.6-sol`, high effort, 300s. Persist and audit both resolved Solver and Human Agent identities. Record the code commit and exact command.
- [ ] Run launcher `--dry-run --cpu-slots 10 --hours 4` and assert exactly 10 RUNNING plans, 7 PENDING plans, 17 distinct context paths/hashes, no `--freeze-verifier`, and exact Human Agent flags.
- [ ] Start the batch under one detached watcher. Confirm exactly one watcher; 10 running jobs have correct context hashes, shipped V0, 14,400-second deadlines, exact resolved Solver/Human Agent model and effort, and Human Session activity (opened/closed or progressing) with no `AgentSystemUnavailable`; 7 pending jobs retain context hashes and null deadlines.
- [ ] After first-wave Human Session activity, scan Human transcripts, durable guidance files and `solver_ws/HUMAN_GUIDANCE.md` with the same forbidden-span detector; assert no full context, long evaluator span, secret source path or context path escaped into the run. Exclude only control-plane `batch.json`, which intentionally stores the host context path/hash.
- [ ] Record and dry-run the implemented `evaluate-results` command for eventual 17-task Final Evaluator replay against the exact accepted packages used by the other arms.
- [ ] Record status and monitoring commands for handoff.
