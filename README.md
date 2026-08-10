# Co-Scientist — Human-Guided Environment Refinement for Automated Research

A minimal, runnable MVP of the design doc's central thesis:

> The bottleneck in automated research is not *solution discovery* (layer 3,
> where AlphaEvolve / SimpleTES / Hyra already work). It is **Proxy Environment
> Construction** (layer 2): building — and *continuously refining* — an
> executable research environment that is **hard to game** and can therefore
> support **long, effective exploration** under **bounded human effort**.

This repo does not try to be a full system. It is the smallest thing that makes
the argument *concrete and measurable*: a real gameable evaluator, a rational
solver that collapses it, a scalable human-proxy that refines it, and the three
metrics the doc hinges on — **Hack Rate**, **Effective Exploration Horizon
(EEH)**, and **Real Performance** — plus the **Reference Horizon** ceiling.

## TL;DR — run it

```bash
pip install -e .           # numpy only
python -m coscientist.experiments.run_curve
pytest -q                  # 6 behavioural tests pin the claims
```

Output:

```
Reference Horizon (oracle V*, perfect env): 161 useful ticks

regime                human  hackrate  collapse   EEH  horizon_gap  realperf
----------------------------------------------------------------------------
no-human                  0      0.62        32    32         0.80     1.780
audit/6                  17      0.00         -   161         0.00     1.780
audit/3                  24      0.00         -   161         0.00     1.780
audit/1                  40      0.00         -   161         0.00     1.780
```

Read it as: **without human refinement the evaluator collapses at tick 32
(short EEH, high hack rate). With a Human-Proxy holding the hidden reference
evaluator, collapse is prevented — the effective horizon reaches the reference
ceiling and the hack rate goes to zero.**

## The three-layer framing (and where this repo sits)

| Layer | Question | Status in the field | Here |
|---|---|---|---|
| 1. Problem Identification | *What is worth studying?* | some work; not our focus | out of scope |
| **2. Proxy Environment Construction** | *How to build/refine a hard-to-game env?* | **under-studied** | **this repo** |
| 3. Solution Discovery | *Given an env, find better solutions* | AlphaEvolve, SimpleTES, Hyra | a stub solver |

## The core objects (and how they map to real systems)

The whole point is that the two systems you already have are the **two halves**
of this vision; the loop *between* them is the contribution. Every class here is
a stripped-down stand-in for a real component and is meant to be swapped out.

| Concept in the doc | Class here | Real system it stands in for |
|---|---|---|
| Proxy Environment | `ProxyEnvironment` | autosciworld task package |
| Proxy evaluator **V** (visible, gameable) | `env.proxy_score` | ASW `verifier.py` |
| Reference evaluator **V\*** (hidden) | `env.reference_score` | ASW `_meta/reference.py` + hidden anchors |
| Editable solution x₀ | `Task.seed_payload` | ASW `baseline.py` |
| Solver (evolutionary best-of-K + notes) | `Solver` | ExplorationHarness (SimpleTES layer-1) |
| **Human Proxy** (dense eval guidance + patches) | `HumanProxy` | *the novel middle — nobody built it* |
| Hack Rate / EEH / Real Perf | `metrics.py` | new measurement apparatus |

### Why the Human Proxy is the crux

The design doc's hardest question is *"how do we Scale Human Effort?"* A real
expert can give dense evaluation guidance but cannot be run across hundreds of
tasks and repeated trials. So we replace them with an agent that **holds a
hidden Reference Evaluator V\*** and answers the solver's queries from V\*'s
real results (`coscientist/human_proxy.py`). Two design decisions taken straight
from the doc's margin notes:

1. **Intervention is restricted to the Evaluation channel.** The Human Proxy
   never proposes solutions; it only judges the environment/evaluator.
2. **It is not a bare yes/no** (too sparse; invites gaming the yes/no signal),
   **and it is not just free-text context** (which the solver could learn to
   talk around). When it detects gaming it emits a concrete **verifier patch**
   that hardens V. *The feedback that changes the world is a patch, not a
   paragraph.* Correspondingly, the solver reacts to the **evaluator**, never to
   the feedback text — see `test_solver_never_reads_feedback_text`.

Every `HumanProxy.query` call is **one unit of human effort**. The experiment
measures outcomes against the number of these calls — this is the doc's
"proxy for human effort" made operational.

## The toy task: a faithful reproduction of the ASW TTV DOF-overfit hack

`coscientist/tasks/ttv_overfit.py` reproduces, in numpy only, the canonical
evaluator-collapse case documented in autosciworld
(`docs/hack_cases/ttv_reduced_chi2_overfit`). We fit `y = trend + Σ sinusoids`
to noisy observations.

* **V (proxy, flawed):** reduced-χ² vs the **noisy observations**, with a DOF
  count that (by default) does **not** debit the sinusoid parameters — exactly
  ASW's defect `dof = N − 2·planets` omitting the free modes. Least-squares
  fitting an ever-larger sinusoid basis drives reduced-χ² **below the noise
  floor** → spuriously high proxy score.
* **V\* (reference, hidden):** reduced-χ² vs the **true signal** with honest DOF.
  Overfitting the noise moves *away* from the truth, so V\* cannot be gamed.

A proxy-greedy solver therefore slides into overfitting and collapses V. The
Human-Proxy patch that closes it — `dof_credit_fraction ↑` — is precisely the
"count every submittable degree of freedom" fix ASW's case study recommends.

## What the experiment shows (and an honest finding)

The doc predicts: *as human interventions increase → Hack Rate ↓, EEH ↑ toward
the Reference Horizon, Real Performance ↑.* The MVP reproduces the **Hack Rate ↓
/ EEH ↑ / gap → 0** relationship.

One finding worth flagging because it is more interesting than a hand-tuned
smooth curve: **collapse here is a threshold phenomenon.** It requires an
*unbounded* exploit gradient (basis growth that always pays under the flawed
DOF). A single well-placed intervention that credits *some* of the mode DOF
removes that gradient — it caps the optimal basis at finite size — so runaway
collapse never starts (`test_single_intervention_removes_runaway_gradient`). You
do not need to perfectly fix the evaluator; you need to kill the unbounded
exploit direction. Whether real tasks collapse as a *threshold* or a *gradient*
(producing the doc's smooth human-effort curve) is an empirical question this
scaffold is built to ask.

## Metrics (`coscientist/metrics.py`)

* **Hack Rate** — sliding-window fraction of candidates where V ≫ V\* (proxy
  overstates real performance by more than a margin).
* **Evaluator collapse** — first tick where windowed Hack Rate exceeds a
  threshold *and* best real performance has stopped advancing. (The doc's
  definition verbatim.)
* **Effective Exploration Horizon (EEH)** — useful ticks before collapse.
* **Real Performance** — best V\* achieved.
* **Reference Horizon** — EEH of an oracle run on a perfect environment; the
  ceiling every method tries to approach. `horizon_gap` is the doc's flagship
  y-axis (plotted vs human-intervention count).

## Layout

```
coscientist/
  types.py            Candidate, EvalResult, Note, Feedback, Step
  environment.py      ProxyEnvironment (V + hidden V*), Task, apply_patch
  human_proxy.py      HumanProxy: dense eval feedback + verifier patches
  solver.py           evolutionary best-of-K loop + human-query trigger
  metrics.py          Hack Rate / collapse / EEH / Reference Horizon
  tasks/ttv_overfit.py  the ASW TTV DOF-overfit toy task
  experiments/run_curve.py  the flagship table
tests/test_mvp.py     6 behavioural tests
```

## How to extend toward the real thing

* **Real solver:** replace `Solver` / `Proposer` with an ExplorationHarness
  chain (LLM/agent generation, RPUCG selection). The `propose(parent, notes,
  feedback, seed) -> [Candidate]` contract is the seam.
* **Real environment:** replace the toy task with an ASW task package; `V` =
  its `verifier.py`, `V*` = its `_meta/reference.py`. `apply_patch` becomes an
  edit to the shipped verifier's guard config (DOF budget, held-out split, …).
* **Real human proxy:** back `HumanProxy` with an LLM that reads V\*'s output +
  traces and emits both dense guidance and a concrete verifier patch. Swap in a
  real domain expert for the small-scale SOTA runs; keep the proxy for the
  scalable dataset.
* **The dataset deliverable:** log every `(x_t, q_t, Feedback, patch)` tuple —
  that is the "Human Proxy Dataset" the doc proposes for others to follow.

---

# The co-evolution demo (`coscientist/demo/`)

`run_curve.py` above is the **single-loop** MVP: solutions evolve, and the
evaluator is only *patched* by a HumanProxy holding a hidden `V*`. The
`demo/` subpackage is the **functionally complete** system the design doc
actually describes — **two evolving loops** driven by real coding **agents**,
with a **human port** arbitrating every change to the evaluator.

```
                 ┌──────────── shared blackboard ────────────┐
                 │  archive (solutions) · evaluator V · log   │
                 └────────────────────────────────────────────┘
   LOOP A (fast)                              LOOP B (slow)
   solution evolves                           evaluator evolves
   ────────────────                           ──────────────────
   SolutionProposer = agent                   EvaluatorProposer = agent
   edits solution.json in a workspace,        reads the gamed verifier + hack
   scored by the CURRENT verifier             evidence, rewrites verifier.py
              ↘                     ↙
             ┌──────── HUMAN PORT ────────┐
             │  review(request) → response │  ← APPROVE / REPLACE / REJECT / GUIDE
             │  the ONLY way V can change  │
             └─────────────────────────────┘
```

**What "evolve" means here is concrete, not metaphorical:**

* A **solution** is `solution.json`; the solution agent edits it in a real
  workspace to score higher under the current verifier. → *solution evolves*
* The **evaluator** is `verifier.py` — a rewritable Python source string run
  out-of-process (`evaluator.py`). The evaluator agent rewrites it to close a
  hole. → *evaluator evolves*
* The **human port** (`human_port.py`) is the single seam where a verifier
  rewrite is ratified. No `V*` is required: when there is no oracle, the human
  *is* the source of truth for "is this evaluator change good?" The `AutoHuman`
  is a scalable stand-in that judges via a hidden `V*`; the `CliHuman` blocks
  and asks a real person. Same protocol — swap freely.

## Run it

```bash
# Fully offline, deterministic (stub proposers + AutoHuman). Always runs, no keys.
python -m coscientist.demo.cli --log run.jsonl

# A real coding agent drives BOTH loops; you arbitrate at the terminal.
python -m coscientist.demo.cli --agent codex       --human cli
python -m coscientist.demo.cli --agent claude-code --human cli

# Real agent, scalable AutoHuman stands in for the human.
python -m coscientist.demo.cli --agent codex --human auto
```

The agent backends (`agent_backend.py`) mirror ExplorationHarness's
`simpletes_agent/backends.py` contract: one headless single-shot session per
proposal, prompt on stdin, the agent writes its output file into the workspace,
the caller reads it back. `--no-agent`/`stub` swaps in a deterministic
in-process handler so the entire control flow is exercised with zero
dependencies — the loop wiring is identical either way.

## What the offline run shows

```
verifier versions    : 2  (0=initial flawed -> 1=hardened via agent)
human reviews         : 1
verifier changes      : 1
best solution modes   : 3
best real  score (V*) : -0.24   (hidden ground truth)
```

Traced through the event log:

* **Before the fix**, the solution loop games verifier v0 — an **18-mode**
  overfit becomes the *best-scoring* solution (`proxy = −0.49`, the top of the
  archive) while its **real** performance is catastrophic (`V* = −5.29`). Proxy
  and truth diverge by +4.8: textbook evaluator gaming.
* The evaluator loop assembles that evidence and the agent **rewrites the
  verifier** to charge every free parameter against the degrees of freedom (the
  ASW-recommended DOF fix).
* The **human port ratifies** the rewrite (v0 → v1). The archive is **rescored**
  under the new V, and the overfit stops paying — the winning solution reverts
  to a genuine **3-mode** fit with good real performance.

`test_verifier_change_requires_the_human_port` pins the load-bearing property:
with a *rejecting* human, the verifier never changes even though it is provably
gamed. The evaluator cannot evolve behind the human's back — which is the whole
point of keeping the interface open.

## Files

```
coscientist/demo/
  taskspec.py       fixed instance + hidden reference V* (agent-invisible)
  evaluator.py      the evolvable verifier: versioned, run out-of-process
  agent_backend.py  codex / claude-code / harbor backends + offline stub
  harbor_runtime.py package a session as a Harbor task, run it, recover output
  workspace.py      the shared file workspace an agent session operates in
  proposers.py      SolutionProposer + EvaluatorProposer (agent-driven)
  human_port.py     HumanPort protocol + AutoHuman + CliHuman
  eventlog.py       append-only JSONL stream (CLI render + dataset export)
  orchestrator.py   interleave the two loops, route every V change through review
  cli.py            `coscientist-demo` entry point
tests/test_demo.py  behavioural tests (offline)
```

# The container runtime (`harbor_runtime.py`)

The local backends (`codex`, `claude-code`) run the agent as a **subprocess with
its internal sandbox off** (`--dangerously-*`). That is only safe with an *outer
boundary*, and it leaves the hidden reference evaluator **V\*** isolated by mere
convention (the code just doesn't write `taskspec` into the workspace). The
`harbor` backend supplies the real boundary: every session runs in a fresh
**Harbor container**, and grading can run in a **separate** container the agent
never touches.

**All three reference benchmarks are the same framework — Harbor** (from the
Terminal-Bench authors). This box's own `harbor-rollout/` drives it (harbor
`0.20.0`); the shapes below are verified against it, not remembered.

| Reference | Container topology | `--isolation` |
|---|---|---|
| **AutoLab** (36 tasks) | agent + grader in **one** container | `shared` |
| **Terminal-Bench 2/3** (Harbor Index / frontierbench) | grader in its **own** container | `separate` |
| **autosciworld** tasks (`asw_000104`) | grader offline, own container | `separate` (default) |

The key seam: **`AgentBackend.run_session(workspace, prompt) → agent writes files
back into workspace` is already the runtime seam.** `HarborBackend` is a third
implementation of that same Protocol — so the proposers and orchestrator change
**zero lines**. One `run_session` = one `harbor run` job = one fresh container,
which is exactly "each role its own container".

In this system Harbor is **only the agent sandbox**; grading V still runs on the
trusted control plane (`Evaluator.run`, out-of-process). The Harbor task's own
verifier is therefore vestigial (`tests/test.sh` writes `reward=0`) — we use
Harbor to (a) isolate the agent and (b) recover the one file it writes
(`solution_out.json` / `verifier_out.py`) through Harbor's declared `artifacts`
channel. What Harbor does *not* give us is a **rewritable** verifier — the whole
Loop B contribution — so we are not a Harbor task; we are an orchestrator that
*drives* Harbor, holding the evolving V on the control plane.

```bash
# each agent session runs in its own container; grader isolated (TB2/TB3 model)
python -m coscientist.demo.cli --agent harbor --isolation separate --human auto

# agent + grader share a container (AutoLab model)
python -m coscientist.demo.cli --agent harbor --isolation shared   --human auto

# site overlay (e.g. a DNS-pin hosts_overlay.yaml), repeatable
python -m coscientist.demo.cli --agent harbor --extra-compose hosts_overlay.yaml
```

`harbor_runtime.py` is **pure and docker-free** — it *builds* the task package
(`task.toml` schema 1.3, `environment/`, vestigial `tests/`) + the JobConfig, and
*parses* a finished job dir (`jobs_dir/<job>/<trial>/artifacts/app/<output>`).
Only the `harbor run` subprocess in `HarborBackend` needs docker; absent it, the
session returns a clean failure instead of crashing. The packaging and parsing
are fully unit-tested offline, and the emitted JobConfig is accepted by real
`harbor run --print-config`.

**V\*-isolation is a hard, tested invariant** — the software stand-in for the
container boundary during local runs. `assert_no_reference_leak` scans every file
about to be packaged for V\* fingerprints (`reference_score`,
`reduced_chi2_vs_truth`, the true-signal samples) and fails closed;
`pack_task` calls it before anything could ship to a container. The `AutoHuman`
legitimately holds V\* but lives on the control plane, never in a task package, so
it is never scanned.



