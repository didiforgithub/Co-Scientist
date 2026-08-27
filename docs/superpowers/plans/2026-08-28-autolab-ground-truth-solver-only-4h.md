# AutoLab Ground-Truth Solver Only 4h Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox tracking and require a spec review followed by a code-quality review for each implementation task.

**Goal:** Launch one auditable 17-task AutoLab arm on A100 with the accepted Final Evaluator fixed from turn zero, no Human Proxy, no verifier evolution, private development seeds, ten concurrent Solver runs, and a four-hour budget per task.

**Architecture:** A dedicated experiment module transactionally generates private seeds and pre-seeds all runs from accepted evaluator artifacts, trusted checker trees, and the original seed solution. The launcher persists `freeze_verifier` as an immutable first-class contract. Static, real-backend smoke, privacy, resource, and runtime-invariance audits guard the experiment before and after launch. Search-time development seeds stay distinct from the fixed held-out replay seed.

**Tech Stack:** Python 3.11, pytest, existing Co-Scientist AgentSystem/launcher, Docker-backed AutoLab checkers, SSH deployment to `Quant-A100-8`.

---

## Task 1: Make frozen-verifier mode a persisted launcher contract

**Files:**
- Modify: `coscientist/coevo/launcher.py`
- Modify: `tests/test_coevo.py`

- [ ] Write failing tests proving `--freeze-verifier` is emitted for every launched child, stored at batch/run/run-contract levels, restored by `_spec_of`, checked by idempotent start/watch/spawn, and preserved for pending promotion and crash respawn.
- [ ] Add `freeze_verifier: bool` to `LaunchSpec` and `Batch.start`; expose public launcher `start --freeze-verifier` and append the child flag only when true.
- [ ] Upgrade the schema deliberately: legacy batches interpret an absent field as `false`; a new frozen batch must carry `true` at top level and in every run and immutable run contract. Missing or contradictory frozen fields fail closed.
- [ ] Prove independent `watch`, pending promotion, and respawn reconstruct the persisted value rather than a CLI default; reject idempotent start `false` against persisted `true` before spawn.
- [ ] Keep the flag protected from `--extra-args` overrides and run focused plus complete non-environmental tests.

## Task 2: Transactionally prepare and statically audit the 17 runs

**Files:**
- Create: `coscientist/experiments/autolab_ground_truth_solver_only.py`
- Create: `tests/test_autolab_ground_truth_solver_only.py`
- Reuse without weakening: `coscientist/experiments/autolab_human_proxy.py`

- [ ] Write failing tests for the exact old10/new7 set, accepted artifact binding, trusted checker selection, original seed solution, fixed accepted v0, and absent Human features.
- [ ] Add `prepare-seeds`: build and audit all 17 cryptographically random per-task seeds under a sibling staging root, publish the whole outside-runs directory once without replacement, use directory `0700` and files `0600`, print no values, and expose only SHA-256 identities.
- [ ] Add `prepare-runs`: reject an existing dedicated root, build all 17 run boundaries under one sibling staging root, audit the complete set, and atomically publish once so a partial experiment is never launcher-visible.
- [ ] Copy each accepted `final_verifier.py` to `bootstrap_ws/verifier.py` and `supervisor/verifier_versions/v0.py`; versions metadata must contain only accepted Final Evaluator v0.
- [ ] Materialize `bootstrap_ws/ctx.json` from the accepted template with the run-local trusted checker, private development seed, and `validation_mode=false`; it is the only allowed run-local secret file and must have restricted permissions.
- [ ] Copy the old10/new7 trusted checker tree and corrected-control seed solution/resources. The Solver brief says normal queries expose feasible + score, exceptional detail is sanitized/audited, and the evaluator is fixed.
- [ ] Bind resources by comparing task `resource.toml` with the trusted checker-run manifest: require identical image/CPU/memory/timeout, `allow_internet=false`, resolve the immutable image ID, and record a canonical contract hash.
- [ ] Write a manifest with `arm=autolab_ground_truth_solver_only`, `freeze_verifier=true`, no Human features, `gpt-5.6-sol/high`, weak topology, 14,400 seconds, accepted/checker/resource hashes, and development seed hash only.
- [ ] `audit-runs` must fail closed on task, symlink, version, accepted/checker/seed/resource, permission, unexpected runtime artifact, Human feature, or resumability drift.
- [ ] Verify the Solver isolation contract: its container mounts only `solver_ws`; neither run root, `bootstrap_ws`, checker, nor seed directory is mounted; `_refresh_solver_ws` never copies `ctx.json`.
- [ ] Implement privacy scanning for raw, hex, base64, URL-safe seed encodings and private paths across solver stdout/stderr, candidates, evaluator error detail, queries, and manifests, allowing the secret only in private seed storage and `bootstrap_ws/ctx.json`.
- [ ] Run focused plus complete non-environmental tests.

## Task 3: Smoke all pre-seeded evaluators on the real backend

**Files:**
- Modify: `coscientist/experiments/autolab_ground_truth_solver_only.py`
- Modify: `tests/test_autolab_ground_truth_solver_only.py`

- [ ] Add `smoke-runs` because `--resume` bypasses the normal Bootstrap V0 smoke.
- [ ] For every task, run its original seed solution once through the accepted Final Evaluator with its actual development seed and run-local checker, using the exact formal immutable image/CPU/memory/timeout and `network=none` contract.
- [ ] Fail on crash, timeout, malformed result, or non-finite score. A clean correctness rejection may pass but records `feasible=false` and its baseline score.
- [ ] Store the hash-only report outside runs, never print/store seeds, and never mutate formal events, queries, candidates, or manifests.
- [ ] Repeat static/privacy audits after smoke. Enforce: `prepare-seeds → prepare-runs → audit-runs → smoke-runs → audit-runs/privacy → launcher dry-run → launch`.

## Task 4: Implement the held-out replay and runtime-invariance audits

**Files:**
- Modify: `coscientist/experiments/autolab_ground_truth_solver_only.py`
- Modify: `tests/test_autolab_ground_truth_solver_only.py`

- [ ] Implement a dedicated replay (or safely generalized existing replay) that accepts only this arm, accepted-final origin, and frozen verifier.
- [ ] Accept `done` and `budget_spent` only with a valid best solution; reject held/max-respawn, missing best, or ambiguous task mapping.
- [ ] Force the fixed held-out audit seed, `repeat=2`, and prior worker/aggregation semantics; prove the audit seed differs from all dev seeds and never appears in search files.
- [ ] Add replay `--dry-run` to validate all 17 evaluator/checker/image/resource contracts before launch without requiring completed candidates.
- [ ] Add live/final invariance auditing: only `v0.py`; bootstrap/v0 match accepted hash; one accepted versions entry; every eval query uses version zero; zero hardenings; frozen batch/run contracts; no mode switch, Human Session, or Human Proxy.

## Task 5: Review implementation and deploy to A100

**Files:**
- Review all Task 1–4 changes against this plan.

- [ ] For every implementation task, run independent spec-compliance review, fix findings, then code-quality review and fix findings.
- [ ] Run syntax, diff, focused, and complete relevant tests with fresh output.
- [ ] Commit on `feat/autolab-ground-truth-solver-only`, push, create `/home/zhangjiayi/github/Co-Scientist-autolab-ground-truth-solver-only`, and verify its HEAD.

## Task 6: Prepare, dry-run, and launch the experiment

**A100 paths:**
- Accepted: `/home/zhangjiayi/github/Co-Scientist/autolab_final_verifier_review_20260825`
- Corrected seed source: `/home/zhangjiayi/github/Co-Scientist/runs/autolab_shippedv0_control17_4h_v2_20260825__autolab_<task>`
- Old10 trusted checker: `/home/zhangjiayi/github/Co-Scientist/runs/autolab_all_4h__autolab_<task>/checker`
- New7 trusted checker: `/home/zhangjiayi/github/Co-Scientist/runs/autolab_proofgate_4h__autolab_<task>/checker`
- Private seeds: `/home/zhangjiayi/github/Co-Scientist-autolab-ground-truth-private-20260828/dev_seeds`
- Runs: `/home/zhangjiayi/github/Co-Scientist-autolab-ground-truth-runs-20260828`
- Batch: `autolab_ground_truth_solver_only_17_4h_20260828`

- [ ] Confirm destinations absent, sources present, Docker healthy, and A100 gateway exactly `gpt-5.6-sol/high`.
- [ ] Generate/audit private seeds, prepare/audit runs, smoke all 17 real backends, repeat privacy/static audits, and run held-out replay `--dry-run`.
- [ ] Launcher dry-run with `--cpu-slots 10 --hours 4 --freeze-verifier --feedback feasible_score --solver-strength weak --solver-model gpt-5.6-sol --solver-reasoning-effort high`; verify ten RUNNING, seven PENDING, no Human flag, frozen flag on all first-wave argv.
- [ ] Start the identical immutable command under a detached watcher.
- [ ] Verify 17 exact contracts, ten live + seven pending, 14,400 seconds each, frozen true, no Human feature, and expected Solver identity.
- [ ] Run live privacy/invariance auditing after workspaces exist and again after each first-wave task has at least one eval query; stop/report rather than relax a failed contract.

## Task 7: Run final held-out comparison only after completion

- [ ] Once all tasks are `done` or valid `budget_spent`, run the dedicated replay with `repeat=2` and `shipped-v0-control17-final-replay-audit-only-20260825-v1`.
- [ ] Re-assert audit seed differs from all development seeds and is absent from every search-time file.
- [ ] Report the launch separately from final comparative scores.
