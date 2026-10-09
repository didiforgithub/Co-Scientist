"""The agent system (§4) — the SYSTEM builds and evolves its own evaluator.

This is the general path the user asked for: drop **raw input** (some files + a
text description — nothing more, NO ProblemSpec) into the system, and the system
itself researches the problem, authors the evaluation, and attempts/advances it.
The orchestrator here is plain code — the hub in a hub-and-spoke topology. It never
understands the problem's content; three coding-agent sessions do all the thinking:

  1. **Bootstrap (Supervisor authors V0).** A one-shot Supervisor container is
     handed the raw input and the ``verify(payload, ctx) -> {feasible, raw,
     artifacts}`` contract. It researches the problem and WRITES the evaluation:
     ``verifier.py`` (the black box V0), ``ctx.json`` (the instance the verifier
     needs), ``seed_solution.json`` (a scorable starting point), and ``probes.json``
     (degenerate/adversarial solutions it KNOWS should score low — the red-team
     contract, generalized from the curve-fit "mode-count monotonicity" oracle).
     The orchestrator recovers these at the boundary as V0. Nothing is presupposed
     about the domain.

  2. **Solve (long-lived Solver container + file-state resume).** The Solver agent
     works against the black box through the ``container-eval`` / ``container-status``
     / ``container-ask-supervisor`` shims (a bind-mounted unix socket). It EXITS when
     it wants to ask the Supervisor; the orchestrator processes the request and a
     fresh ``claude -p`` in the SAME open container resumes from file state + the
     agent's scratchpad. No polling.

  3. **Evolve (Supervisor agent-authored detect + harden).** On a review request —
     or on the orchestrator's own wall-clock cadence — a one-shot Supervisor
     container is given the current V source, the solver's best solution, and the
     probe evidence, and asked to decide whether V is being gamed and, if so, to
     REWRITE a strictly-harder V that still scores the seed feasibly. Detection AND
     hardening are agent-authored, because a canned curve-fit fix would overwrite
     the problem's own verifier (the load-bearing consequence).

The evaluator mechanism (``demo.evaluator.Evaluator``), the black-box wrapper
(``EvalService``), the boundary store (``RunStore``), and the wall clock
(``Deadline``) are reused unchanged — they were already domain-agnostic. Only the
verifier *source strings* and the *ctx/solution shapes* are agent-authored now,
instead of the hardcoded curve-fit ones.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Protocol

from ..demo.evaluator import Evaluator
from . import resources as _resources
from .budget import Deadline
from .container import (
    ControlSocket,
    DockerContainer,
    GatewayConfig,
    agent_elf_path,
    docker_unavailable,
    one_shot_agent,
)
from .eval_service import EvalService, FeedbackLevel
from .human_sessions import SessionOutcome
from .store import RunStore


# ---------------------------------------------------------------------------
# recovered bootstrap artifacts
# ---------------------------------------------------------------------------
@dataclass
class Bootstrap:
    """What the Supervisor bootstrap agent authored, recovered at the boundary."""

    verifier_src: str
    ctx: dict
    seed_solution: dict
    probes: list[dict]          # each: {"description": str, "solution": {...}}
    notes: str = ""
    # Optional agent-authored feedback module (free-form disclosure strategy). None =
    # the plain numeric disclosure (legacy FeedbackLevel path). See evaluator.run_feedback.
    feedback_src: Optional[str] = None
    # Optional solver container overrides the bootstrap agent requested (image/gpus),
    # e.g. a CUDA/kernel task. Empty = use the system defaults.
    solver_env: dict = field(default_factory=dict)
    # Cold judgement (from reframe_policy.json): does THIS problem admit a rigorous
    # proof/disproof reframing at all? Optimization tasks (kernel/CPU/throughput/latency)
    # have only a practical ceiling and get False — they may NEVER be switched to a proof
    # game. Math/physics/optimality-or-bound tasks get True plus the concrete claim.
    admits_proof: bool = False
    provable_claim: str = ""


class AgentSystemUnavailable(RuntimeError):
    """Raised when docker or the agent binary is absent — a clean, explained stop."""


class HumanInteractionPort(Protocol):
    """Blocking checkpoint used by live Feishu and deterministic tests."""

    def consult(self, *, purpose: str, context: dict) -> Optional[SessionOutcome]: ...


# ---------------------------------------------------------------------------
# prompts — domain-agnostic; the raw input carries all the problem content
# ---------------------------------------------------------------------------
_VERIFY_CONTRACT = '''\
The evaluation you write is a Python module `verifier.py` exposing exactly:

    def verify(payload: dict, ctx: dict) -> dict:
        # payload: a candidate solution (a dict — you define its shape)
        # ctx:     the fixed instance data your verifier needs (from ctx.json)
        return {"feasible": bool, "raw": float, "artifacts": {...}}
        # `raw` is the score: HIGHER IS BETTER. `artifacts` is any diagnostics
        # you choose to expose. It must be pure-Python stdlib (no third-party
        # imports), deterministic, and finish in a few seconds.

It is run OUT OF PROCESS on untrusted input, so it must never hang or crash on a
malformed payload — guard everything and return feasible=False with raw very low
for garbage input.'''

_BOOTSTRAP_PROMPT = '''\
You are the SUPERVISOR in a co-evolution system. Raw input describing a problem has
been dropped into this directory (/work). Read EVERYTHING here first (there may be a
text description and/or other files), reason about the problem from the input and
your own knowledge, and decide how a candidate solution to it should be JUDGED.

Work primarily from the raw input and your own knowledge; do not block on external
research. Encode your best-judgment model directly into the verifier. Use Read/Write
and shell (python3) on local files.

IF THE RAW INPUT ALREADY SHIPS A RUNNABLE CHECKER (e.g. an `environment/` folder with
a reference model + scorer, an oracle, a grader, or an `evaluate.py`): do NOT reinvent
it. WRAP it — your `verifier.py` should import or shell out to the provided checker and
map its pass/fail + primary metric onto `{{feasible, raw, artifacts}}` (raw higher =
better; e.g. a speedup or accuracy). Put the checker's files under a `checker/` folder
in /work; at run time they will be available on the host at the absolute path in
`ctx["checker_dir"]`, so read them via `ctx["checker_dir"]` and NEVER hardcode a path.
Then encode the checker's anti-gaming rules (correctness on hidden/randomized inputs,
final-state checks, no shortcuts) as `probes.json` degenerate cases.

If your verifier compiles or runs the candidate in a subprocess (e.g. `make`, `gcc`,
a test binary), you MAY bound it with `RLIMIT_CPU`, `RLIMIT_AS`, `RLIMIT_FSIZE`,
`RLIMIT_CORE` in a `preexec_fn`. But do NOT set `RLIMIT_NPROC`: the verifier already
runs inside an isolated, process-capped container, and this container shares its host
UID, so an in-process `RLIMIT_NPROC` is counted per-UID across the whole host and will
make `fork()`/`exec` fail with "Operation not permitted" (a false all-infeasible). Rely
on the container's own process cap, not an rlimit, to contain fork bombs. Likewise, if
you cap how many bytes you read from a subprocess's stdout, that cap MUST exceed the
LARGEST output any legitimate case produces (e.g. a hex dump of your biggest hidden
input) — a read cap smaller than a correct candidate's own output silently truncates it
and rejects every honest solution. Prove it: actually run your verifier's FULL path
(including its largest hidden/benchmark case) on the seed and confirm it is feasible
before you finish — an infeasible seed means the whole run has no baseline to build on.

Then AUTHOR THE EVALUATION. Write these files into /work:

1. `verifier.py` — {contract}

2. `ctx.json` — a JSON object with whatever fixed data your verifier reads from
   `ctx` (e.g. an instance, target constants, parameters). May be {{}} if none. If you
   wrapped a bundled checker, you may reference `ctx["checker_dir"]` (the orchestrator
   fills in its host-absolute path — leave your own placeholder value, it is overwritten).

3. `seed_solution.json` — one concrete, valid candidate solution (in the payload
   shape your verifier expects) that a solver can score immediately as a baseline.
   It should be feasible but NOT good — a starting point.

4. `probes.json` — a JSON list of adversarial/degenerate candidates you believe a
   GOOD verifier should score LOW, each as {{"description": "...", "solution": {{...}}}}.
   These are the red-team: if a weak verifier scores any of them competitively high,
   it is being gamed. Include at least 2 that try to exploit obvious shortcuts.

5. `SOLVER_BRIEF.md` — a short, self-contained brief telling a solver agent what the
   problem is, the exact payload shape to produce, and that it is scored by a hidden
   verifier via `./container-eval <solution.json>`. Do NOT reveal verifier.py's
   internals in this brief.

6. `feedback.py` (OPTIONAL) — a free-form disclosure strategy. If present it exposes:
       def feedback(payload, ctx, verify_result, history) -> dict
   returning {{"detail": str, "artifacts": {{...}}}} — whatever qualitative or
   diagnostic guidance the solver should see THIS query (the verifier's numeric score
   and feasibility stay authoritative and are added by the system). `history` is a
   bounded tail of prior scores. It is stdlib-only and may `import llm_client` (a
   provided OpenAI-style `chat(messages, *, model, temperature, timeout)` +
   `available()`) to consult an LLM for natural-language guidance. Omit this file to
   use the plain numeric disclosure — only add it if guided feedback helps this problem.

7. `solver_env.json` (OPTIONAL) — {{"image": "<docker image>", "gpus": "<spec>"}} if the
   SOLVER needs a specific container image or GPU access to develop/run candidates
   (e.g. a CUDA/kernel task). Omit for a plain python problem.

8. `reframe_policy.json` — a COLD, up-front judgement about the NATURE of this problem:
   does it admit a rigorous PROOF/DISPROOF at all, as opposed to only a practical
   optimization ceiling? Decide from the problem's intrinsic type, NOT from any future
   solver progress:
     * OPTIMIZATION / ENGINEERING tasks — kernel / CPU / GPU / throughput / latency /
       memory-bandwidth / compression-ratio / "make X faster or smaller" — have only an
       empirical ceiling. There is no theorem to prove; a flat score just means the
       current implementation is near its practical limit. These get
       `{{"admits_proof": false}}` and MUST NEVER be reframed into a proof game.
     * MATH / PHYSICS / ALGORITHMIC-OPTIMALITY-OR-BOUND tasks — where a specific claim
       could in principle be proved or refuted (an optimal value, a lower/upper bound, an
       impossibility, a closed form, an invariant) — get `{{"admits_proof": true}}` PLUS
       `"provable_claim"`: the concrete statement that could be proved/disproved (e.g.
       "no 12-input sorting network uses fewer than 39 comparators"). This does NOT force
       a proof — it only records that a proof reframing is admissible IF the solver later
       genuinely exhausts construction and reaches the theoretical limit.
   Shape: {{"admits_proof": bool, "provable_claim": "<claim or ''>", "reasoning": "..."}}.
   When unsure, default to `false` — a wrongly-allowed proof switch derails an
   optimization run, while a wrongly-forbidden one merely keeps a math task in
   construction (still a valid game).

Verify your verifier.py actually imports and runs on seed_solution.json and on each
probe before you finish (run it with python3); if you wrote feedback.py, run it too.
This is the ENTIRE evaluation the rest of the system will use — be rigorous. When done,
write a one-line `BOOTSTRAP_DONE` marker file.'''

# The harden prompt is the Supervisor's creative brief. It carries TIGHTEN and EXPOSE-MORE
# always; the SWITCH/REFRAME option (c) is injected ONLY when the problem was cold-judged
# to admit a proof/disproof (reframe_policy.json → admits_proof). The orchestrator does NOT
# decide "plateau"; the Supervisor owns the evaluation and judges for itself whether the
# current game is mined out. Its default bias: while wall-clock remains, bite the current
# game HARDER — a plateau is treated as a construction ceiling to push, never an automatic
# cue to switch games. Reframe (when even available) is a high-bar last resort.
_HARDEN_PROMPT = '''\
You are the SUPERVISOR. You OWN THE EVALUATION for a problem — the hidden verifier V, an
optional free-form feedback module, AND the very GAME the solver plays (what "solving"
means: maximize a number, or build a rigorous argument, or disprove a claim, ...). A
solver has been optimizing against your evaluation and may be GAMING it (exploiting a
flaw to score high without truly solving the problem), STUCK (needs more information to
progress), or genuinely near the ceiling of the current game. Be CREATIVE — you are not
a one-way anti-hack ratchet and you are not locked into forward construction.

In /work you have:
  * `problem/` — the original raw problem input (read it to recall what a real
    solution means),
  * `current_verifier.py` — the current V source,
  * `current_feedback.py` — the current feedback module, if one exists,
  * `best_solution.json` — the solver's current best candidate,
  * `probe_report.json` — your red-team probes and the score each got under current
    V (a probe scoring competitively high is evidence of a hole),
  * `progress.json` — CONTEXT for your judgment, not a verdict: the recent score
    trajectory, how much wall-clock the CURRENT game has already burned, how much time
    remains, the current mode, and the turn count. YOU decide from this whether the
    current game still has juice or is mined out. Do NOT treat a flat stretch as an
    automatic signal to switch — a flat score often just means "keep biting harder".
  * `solver_scratchpad.md` / `solver_review_request.json` — the solver's OWN log of what
    it observed while running against V (present only if it wrote them). READ THEM FIRST.

CRITICAL — distinguish a VERIFIER weakness from a HARNESS/INFRASTRUCTURE fault. If the
solver's notes or V's own artifacts show that EVERY candidate scores 0 / infeasible for
reasons UNRELATED to solution quality — e.g. a build cache or permission error
(`mkdir /.cache: permission denied`), a missing toolchain, an unwritable directory, a
crash BEFORE the candidate is even evaluated — then the fault is in HOW V executes, not
in how strictly it judges. In that case DO NOT tighten and DO NOT reframe the game:
rewrite `verifier.py` to fix the execution environment (for example set an explicit,
writable cache such as `GOCACHE`/`HOME`/`XDG_CACHE_HOME` under a tempdir before invoking
the toolchain, create needed dirs, or otherwise make the run succeed). A harness fault
that suppresses all scores must be repaired, never mistaken for solver gaming.

You may evolve the evaluation in ANY combination of these directions:
  (a) TIGHTEN — rewrite `verifier.py` so exploit/probe solutions score strictly LOWER
      while a genuine solution still scores well (close a gaming hole);
  (b) EXPOSE MORE — write/adjust `feedback.py` to hand the solver richer diagnostics or
      guidance when it is honestly stuck (the numeric score stays authoritative);{reframe_option}

DEFAULT BIAS: while wall-clock remains, keep biting the CURRENT game harder (tighten /
expose-more / demand a sharper candidate). A flat score is NOT a signal to change the
game — for most problems it means the current implementation is near its practical
ceiling and the solver should push the construction further, not abandon it.{reframe_bias}

Decide and write into /work:

1. `verdict.json` — {{"gaming": bool, "ask_human": bool, "reasoning": "..."}}.
   Set `ask_human` to true ONLY when this proposed change needs external human/proxy
   calibration (for example an ambiguous contract, a high-impact mode change, or
   evidence that you cannot resolve from the bounded run artifacts). Routine anti-gaming
   hardening with a clear contract should set it to false. Omitting it is equivalent to
   false for backwards compatibility.

2. If you change the verifier, write `verifier.py` — same contract as before:
   def verify(payload, ctx) -> {{"feasible","raw","artifacts"}}, stdlib-only, robust to
   garbage. `seed_solution.json` is only a starting candidate, not a trusted reference;
   do not preserve its score merely to satisfy a gate. The orchestrator only smoke-tests
   that your verifier runs safely on the seed and probes. Semantic quality is reviewed
   by the Supervisor and Human/Proxy. If V needs no change, do not write it.

3. If richer/guiding disclosure helps, write `feedback.py`:
       def feedback(payload, ctx, verify_result, history) -> dict  # {{"detail","artifacts"}}
   stdlib-only, may `import llm_client` for LLM guidance. Omit to keep current disclosure.
{reframe_output}
Verify whatever you write runs (python3) on the seed and probes before finishing.

Write a one-line `HARDEN_DONE` marker when finished.'''

# The (c) reframe option and its output spec are injected ONLY when the problem was
# cold-judged to ADMIT a proof/disproof (reframe_policy.json admits_proof=true). For
# optimization/engineering tasks the reframe option is withheld entirely — there is no
# theorem to prove, so a plateau must be attacked as construction, never escaped into a
# subjective proof game.
_REFRAME_OPTION_ALLOWED = '''
  (c) SWITCH / REFRAME THE REPRESENTATION — available for THIS problem because it was
      cold-judged to admit a rigorous argument (see the provable claim below). This is a
      LAST RESORT, not a default escape hatch, and it is NOT triggered by a plateau. Use
      it ONLY when ALL of the following hold, and say so explicitly in your reasoning:
        (i) the solver has genuinely EXHAUSTED forward construction — many distinct,
            serious attempts, not merely a flat stretch;
        (ii) the best candidate is at what is credibly the THEORETICAL limit of the
             problem (not just the current implementation's practical ceiling); and
        (iii) there is a SPECIFIC provable/refutable claim to attack, namely:
             "{provable_claim}"
      If any of the three fails, do NOT switch — keep biting construction. Reframing an
      optimization plateau into a proof game is a gaming failure, not progress.'''

_REFRAME_BIAS_ALLOWED = ''' Only reframe (c) when its three conditions genuinely hold —
the construction route is exhausted AT THE THEORETICAL LIMIT and a concrete provable
claim remains. Reframing early throws away a game that still had room; your judgment,
not a threshold, and the bar is high.'''

_REFRAME_OUTPUT_ALLOWED = '''
4. If — and only if — conditions (i)-(iii) above genuinely hold and you choose to SWITCH /
   REFRAME the representation (c), ALSO write:
   * `mode_switch.json` — {{"switch": true, "to_mode": "<name, e.g. proof or disproof>",
     "justification": {{"construction_exhausted": "...", "at_theoretical_limit": "...",
     "claim": "..."}}, "reasoning": "..."}}
   * a NEW `verifier.py` IN THE NEW REPRESENTATION. When the new game is qualitative
     (proof/disproof), make it an LLM-VERIFIER: `import llm_client`, ask the model to
     assess the argument's rigor/progress, map that onto `raw` (higher = closer to a
     real proof / a valid refutation). Degrade gracefully (finite low `raw`) when
     `llm_client.available()` is False.
   * `feedback.py` — natural-language guidance in `detail` (via `llm_client`) steering
     the solver in the new game.
   * a NEW `seed_solution.json`, `probes.json`, and `SOLVER_BRIEF.md` IN THE NEW
     REPRESENTATION (seed = a minimal honest argument; probes = hand-wavy / circular
     attempts that MUST score LOW). The system re-baselines the solver on the new game.
'''

# Withheld case: the problem is optimization/engineering — no proof reframing exists.
_REFRAME_OPTION_WITHHELD = ''
_REFRAME_BIAS_WITHHELD = (
    ' This problem admits NO proof/disproof reframing (it is an optimization/engineering '
    'task with only an empirical ceiling); switching the game is NOT an available move. '
    'Your only tools are TIGHTEN and EXPOSE MORE — attack any plateau as construction.'
)
_REFRAME_OUTPUT_WITHHELD = ''


# ---------------------------------------------------------------------------
# the orchestrator
# ---------------------------------------------------------------------------
@dataclass
class AgentSystem:
    """Plain-code hub: builds the agent-authored evaluator and runs the two loops.

    ``raw_input_dir`` is the drop-in (files + text). ``run_dir`` is the RunStore
    root. ``budget_s`` is the wall-clock ceiling for the whole run.
    """

    raw_input_dir: Path
    run_dir: Path
    budget_s: float = 1800.0
    image: str = "python:3.11-slim"
    feedback_level: FeedbackLevel = FeedbackLevel.WITH_ARTIFACTS
    bootstrap_timeout_s: float = 900.0
    harden_timeout_s: float = 600.0
    # A harden is only meaningful if the Solver then RE-BASELINES under the new V.
    # We keep a protected slice for that guaranteed post-harden solve turn, and a
    # floor so a budget-squeezed normal turn still gets usable time. Normal turns
    # are capped to leave (harden_timeout_s + post_harden_solve_s) in reserve, so
    # even the LAST review before the deadline can still harden AND re-solve.
    post_harden_solve_s: float = 900.0
    min_solver_turn_s: float = 60.0

    # Control-arm switch. When True, the Supervisor never hardens/evolves the verifier:
    # bootstrap still authors V0, the Solver still mines it for the full budget, but
    # _run_supervisor_harden early-returns so V is frozen at v0 (hardenings=0). This is
    # the "solve a weak evaluator" arm that pairs against the evolving treatment arm.
    freeze_verifier: bool = False

    # Per-problem LLM creds for HOST-SIDE eval/feedback code that consults a model
    # (e.g. an LLM-verifier). A plain dict {"api_key","base_url","model"}; provided by
    # the operator/init. Injected ONLY into the host-side Evaluator subprocess (via
    # eval_service._eval_env) as LLM_API_KEY/LLM_BASE_URL/LLM_MODEL — NEVER into the
    # Solver container. Never written to disk; events record only its presence.
    llm_config: dict = field(default_factory=dict)
    # Operator/bootstrap overrides for the long-lived Solver container. Default None =
    # use ``image`` and no GPU. A kernel/CUDA task sets these (also via solver_env.json).
    solver_image: Optional[str] = None
    solver_gpus: Optional[str] = None
    # First-class resource config with two ISOLATED slices (solver / verifier). Sourced
    # by layering builtin defaults < resource.toml (loaded from raw_input_dir) < CLI
    # overrides folded in here. Default empty ⇒ current behavior everywhere: a plain
    # Solver container and host-subprocess verify. ``resource_config_path`` is an
    # explicit --resource-config that wins over auto-discovery. ``resource_overrides``
    # is a ResourceSpec whose non-None fields override the loaded file (CLI wins).
    resource_config_path: Optional[Path] = None
    resource_overrides: Optional["_resources.ResourceSpec"] = None

    gateway: Optional[GatewayConfig] = None
    agent_elf: Optional[Path] = None

    # Optional scarce human-expert channel. Tests/custom deployments inject a port
    # directly; CLI preflight builds either the Feishu port or a model-backed Proxy
    # that receives private evaluator text but has no evaluator execution path.
    # The hard maximum is intentionally not configurable: one Co run gets five
    # sessions, while each session may have arbitrarily many natural-language turns.
    human_port: Optional[HumanInteractionPort] = None
    human_expert_id: Optional[str] = None
    lark_cli_executable: str = "lark-cli"
    human_agent_timeout_s: float = 180.0
    human_agent_model: str = "gpt-5.6-luna"
    human_agent_reasoning_effort: str = "low"
    human_proxy_context_path: Optional[Path] = None

    store: RunStore = field(init=False)
    deadline: Deadline = field(init=False)
    evaluator: Optional[Evaluator] = field(default=None, init=False)
    eval_service: Optional[EvalService] = field(default=None, init=False)
    _ctx: dict = field(default_factory=dict, init=False)
    _seed: dict = field(default_factory=dict, init=False)
    _probes: list = field(default_factory=list, init=False)
    _best_payload: Optional[dict] = field(default=None, init=False)
    _best_score: float = field(default=float("-inf"), init=False)
    hardenings: int = field(default=0, init=False)
    reviews_handled: int = field(default=0, init=False)
    # The evaluation's current GAME. Starts "construction" (optimize a scalar); the
    # Supervisor may REFRAME it (e.g. to "proof": an LLM-verifier scores a structural
    # argument). Recovered on resume by scanning events for mode_switch.
    _mode: str = field(default="construction", init=False)
    # Rolling tail of the best score after each extract — CONTEXT surfaced to the
    # Supervisor in progress.json (the orchestrator no longer computes "plateau").
    _score_trajectory: list = field(default_factory=list, init=False)
    # Absolute host path to a bundled checker (materialized under run_dir/checker/ when
    # the raw input ships one), injected into ctx["checker_dir"] so host-side verify can
    # reach it. Empty when the problem ships no runnable checker.
    _checker_dir: Optional[str] = field(default=None, init=False)
    # The resolved two-slice resource spec (defaults < resource.toml < CLI overrides).
    # Populated by ``_resolve_resources`` at bootstrap/resume. Drives the Solver
    # container (solver slice) and the isolated verifier backend (verifier slice).
    resource_spec: "_resources.ResourceSpec" = field(
        default_factory=lambda: _resources.ResourceSpec.defaults(), init=False)
    # set True whenever a harden moves V; the loop then OWES one solve turn so the
    # Solver re-baselines under the new verifier (a co-evolution round is only
    # complete as solve -> harden -> solve-again). Cleared once that turn runs.
    _owe_post_harden_solve: bool = field(default=False, init=False)
    _post_harden_solves: int = field(default=0, init=False)
    # set True when a harden switched the modality; the owed post-harden turn must
    # first re-seed the solver workspace with the NEW contract before it runs.
    _pending_mode_refresh: bool = field(default=False, init=False)
    # wall-clock (deadline.elapsed()) at which the CURRENT game/mode began — reset on
    # every modality switch. Feeds progress.json so the Supervisor can judge how long
    # the current game has been mined without the orchestrator deciding "plateau".
    _mode_since_s: float = field(default=0.0, init=False)
    # monotonically increasing normal-turn counter, surfaced in progress.json.
    _turn: int = field(default=0, init=False)
    # cold admissibility judgement from bootstrap's reframe_policy.json: does this problem
    # admit a proof/disproof reframing at all? Optimization tasks are False (the harden
    # prompt then withholds the SWITCH option entirely). Defaults False (conservative:
    # runs without the file, or unreadable policy, never get the proof escape hatch).
    _admits_proof: bool = field(default=False, init=False)
    _provable_claim: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.raw_input_dir = Path(self.raw_input_dir).resolve()
        self.run_dir = Path(self.run_dir).resolve()
        self.store = RunStore(self.run_dir)
        self.deadline = Deadline(budget_s=self.budget_s)

    # -- preflight --------------------------------------------------------
    def preflight(self) -> None:
        reason = docker_unavailable()
        if reason:
            raise AgentSystemUnavailable(reason)
        elf = self.agent_elf or agent_elf_path()
        if elf is None:
            raise AgentSystemUnavailable(
                "codex agent binary not found (need the @openai/codex npm package "
                "with its vendored static ELF)")
        self.agent_elf = elf
        gw = self.gateway or GatewayConfig.from_host()
        if gw is None:
            raise AgentSystemUnavailable(
                "no codex auth found (need ~/.codex/auth.json)")
        self.gateway = gw
        human_modes = int(bool(self.human_expert_id)) + int(
            bool(self.human_proxy_context_path)
        )
        if human_modes > 1:
            raise AgentSystemUnavailable(
                "choose either a Feishu human or a Human Proxy agent, not both"
            )
        if human_modes and self.human_port is None:
            from .feishu_human import FeishuHumanSessionService
            from .human_evidence import CodexEvidenceAgent
            from .human_sessions import HumanSessionStore

            human_store = HumanSessionStore(self.run_dir, max_sessions=5)
            human_gateway = replace(
                self.gateway,
                model=self.human_agent_model,
                reasoning_effort=self.human_agent_reasoning_effort,
            )
            evidence_agent = CodexEvidenceAgent(
                self.run_dir,
                self.raw_input_dir,
                gateway=human_gateway,
                agent_elf=self.agent_elf,
                image=self.image,
                timeout_s=self.human_agent_timeout_s,
            )
            if self.human_expert_id:
                if not self.human_expert_id.startswith("ou_"):
                    raise AgentSystemUnavailable(
                        "--feishu-expert-id must be a Feishu open_id beginning with ou_"
                    )
                from .feishu_human import BlockingFeishuHumanPort
                from .feishu_transport import LarkCliTransport

                transport = LarkCliTransport(executable=self.lark_cli_executable)
                service = FeishuHumanSessionService(
                    store=human_store,
                    transport=transport,
                    agent=evidence_agent,
                )
                self.human_port = BlockingFeishuHumanPort(
                    service=service,
                    transport=transport,
                    expert_id=self.human_expert_id,
                )
                return

            from .human_proxy_sessions import (
                HumanProxySessionPort,
                LoopbackTransport,
                ModelBackedHumanProxyAgent,
            )

            try:
                proxy_context_path = Path(self.human_proxy_context_path).resolve()
                proxy_context = proxy_context_path.read_text(encoding="utf-8")
                proxy_agent = ModelBackedHumanProxyAgent(
                    evaluator_context=proxy_context,
                    gateway=human_gateway,
                    agent_elf=self.agent_elf,
                    image=self.image,
                    timeout_s=self.human_agent_timeout_s,
                )
            except Exception as exc:  # noqa: BLE001 - private context stays opaque
                raise AgentSystemUnavailable(
                    f"could not initialize Human Proxy context: {type(exc).__name__}"
                ) from exc
            transport = LoopbackTransport()
            service = FeishuHumanSessionService(
                store=human_store,
                transport=transport,
                agent=evidence_agent,
            )
            self.human_port = HumanProxySessionPort(
                service=service,
                proxy_agent=proxy_agent,
                expert_id="human_proxy_agent",
            )

    def _consult_human(
        self,
        *,
        purpose: str,
        context: dict,
        checkpoint: bool = False,
    ) -> Optional[SessionOutcome]:
        """Block at a human checkpoint, persist guidance, and resume safely.

        A completed named checkpoint is replayed from disk on resume without opening
        another scarce session. A Feishu or Proxy session blocks inside ``consult``;
        no evaluator mutation can occur until the expert confirms close.
        """
        checkpoint_path = self.run_dir / "human" / "checkpoints" / f"{purpose}.json"
        if checkpoint and checkpoint_path.is_file():
            try:
                return SessionOutcome.from_dict(
                    json.loads(checkpoint_path.read_text(encoding="utf-8"))
                )
            except (OSError, json.JSONDecodeError):
                pass
        if self.human_port is None:
            return None
        self.store.event("human_session_requested", purpose=purpose)
        outcome = self.human_port.consult(purpose=purpose, context=context)
        if outcome is None:
            self.store.event(
                "human_session_skipped",
                purpose=purpose,
                reason="session_budget_exhausted_or_unavailable",
            )
            return None
        self._persist_human_guidance(purpose, outcome)
        if checkpoint:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(outcome.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary.replace(checkpoint_path)
        self.store.event(
            "human_session_closed",
            purpose=purpose,
            decision=outcome.decision,
            unresolved=len(outcome.unresolved_questions),
        )
        return outcome

    def _persist_human_guidance(
        self, purpose: str, outcome: SessionOutcome
    ) -> None:
        human_dir = self.run_dir / "human"
        human_dir.mkdir(parents=True, exist_ok=True)
        with (human_dir / "guidance.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"purpose": purpose, **outcome.to_dict()}, ensure_ascii=False
                )
                + "\n"
            )
        lines = [f"\n## Human Session: {purpose}\n", f"Decision: {outcome.decision}\n"]
        labels = (
            ("Task contract updates", outcome.task_contract_updates),
            ("New risks", outcome.new_risks),
            ("Approved changes", outcome.approved_changes),
            ("Rejected changes", outcome.rejected_changes),
            ("Human guidance", outcome.human_guidance),
            ("Evidence requested", outcome.evidence_requested),
            ("Evidence generated", outcome.evidence_generated),
            ("Unresolved questions", outcome.unresolved_questions),
        )
        for label, values in labels:
            if values:
                lines.append(f"\n{label}:\n")
                lines.extend(f"- {value}\n" for value in values)
        if outcome.human_rationale:
            lines.append(f"\nHuman rationale:\n{outcome.human_rationale}\n")
        with (self.run_dir / "human_guidance.md").open("a", encoding="utf-8") as stream:
            stream.writelines(lines)

    # ================================================================
    # LOOP 0 — bootstrap: the Supervisor agent authors the evaluator
    # ================================================================
    def bootstrap(self) -> Bootstrap:
        assert self.gateway and self.agent_elf
        ws = self.run_dir / "bootstrap_ws"
        ws.mkdir(parents=True, exist_ok=True)
        # drop the raw input straight into the Supervisor's workspace.
        _copy_tree(self.raw_input_dir, ws)
        human_guidance = self.run_dir / "human_guidance.md"
        if human_guidance.is_file():
            (ws / "HUMAN_GUIDANCE.md").write_text(
                human_guidance.read_text(encoding="utf-8"), encoding="utf-8"
            )
        (ws / "VERIFY_CONTRACT.txt").write_text(_VERIFY_CONTRACT, encoding="utf-8")

        self.store.event("bootstrap_start", raw_input=str(self.raw_input_dir))
        prompt = _BOOTSTRAP_PROMPT.format(contract=_VERIFY_CONTRACT)
        remaining = self.deadline.remaining()
        session = one_shot_agent(
            ws, self.gateway, self.agent_elf, prompt,
            timeout_s=min(self.bootstrap_timeout_s, remaining), image=self.image)
        self.store.cost(who="supervisor", kind="bootstrap_session", calls=1,
                        seconds=0.0, ok=session.ok)

        bs = self._recover_bootstrap(ws, session.note, session.stderr)
        # commit V0 through the reused, domain-agnostic evaluator.
        self.evaluator = Evaluator()
        self.evaluator.versions.append(_version0(bs.verifier_src, bs.feedback_src))
        # bs.ctx already carries ctx["checker_dir"] (pinned in _recover_bootstrap).
        self._ctx, self._seed, self._probes = bs.ctx, bs.seed_solution, bs.probes
        self._admits_proof, self._provable_claim = bs.admits_proof, bs.provable_claim
        # Resolve the two-slice resource spec (defaults < resource.toml < CLI), then
        # apply the agent-authored solver_env.json as a LAST-RESORT fallback — the
        # structured file wins per the user's decision.
        self._resolve_resources()
        self._apply_solver_env_fallback(bs.solver_env)
        self.eval_service = EvalService(
            evaluator=self.evaluator, ctx_provider=lambda: self._ctx,
            feedback_level=self.feedback_level, _eval_env=dict(self._llm_env()),
            _verify_backend=self._verify_backend())
        self.eval_service.on_query = lambda rec, res: self.store.query(_query_line(rec))
        self.store.verifier_version(0, bs.verifier_src, origin="agent",
                                    note="supervisor bootstrap", rationale=bs.notes[:400],
                                    feedback_src=bs.feedback_src)
        self.store.event("bootstrap_done", n_probes=len(bs.probes),
                         ctx_keys=sorted(bs.ctx.keys()),
                         has_feedback=bs.feedback_src is not None,
                         solver_image=self.solver_image, solver_gpus=self.solver_gpus,
                         checker_dir=self._checker_dir,
                         seed_keys=sorted(bs.seed_solution.keys())
                         if isinstance(bs.seed_solution, dict) else [])
        return bs

    def _resolve_resources(self) -> None:
        """Resolve the two-slice ResourceSpec: defaults < resource.toml < CLI overrides.

        Loaded once at bootstrap/resume from ``raw_input_dir`` (or an explicit
        ``resource_config_path``), then CLI overrides folded on top. The legacy
        ``solver_image``/``solver_gpus`` also fold into the solver slice here so the
        old CLI flags keep working. The agent-authored ``solver_env.json`` is applied
        later as a LAST-RESORT fallback (only for still-unset fields)."""
        spec = _resources.load(self.raw_input_dir, explicit_path=self.resource_config_path)
        if self.resource_overrides is not None:
            spec = spec.merge_overrides(solver=self.resource_overrides.solver,
                                        verifier=self.resource_overrides.verifier)
        # legacy solver_image/solver_gpus flags fold into the solver slice (CLI wins).
        legacy = _resources.ContainerResources(
            image=self.solver_image, gpus=_resources._as_gpus(self.solver_gpus))
        spec = spec.merge_overrides(solver=legacy)
        self.resource_spec = spec

    def _apply_solver_env_fallback(self, solver_env: dict) -> None:
        """Fold the agent-authored solver_env.json as a LAST-RESORT into the solver slice.

        The structured resource.toml/CLI wins; solver_env only fills a field still
        unset after ``_resolve_resources``. Also keeps ``solver_image``/``solver_gpus``
        in sync for the event log + container build."""
        if not isinstance(solver_env, dict):
            return
        sol = self.resource_spec.solver
        img = sol.image if sol.image is not None else (
            str(solver_env["image"]) if solver_env.get("image") else None)
        gpus = sol.gpus if sol.gpus is not None else _resources._as_gpus(solver_env.get("gpus"))
        self.resource_spec = self.resource_spec.merge_overrides(
            solver=_resources.ContainerResources(image=img, gpus=gpus))
        self.solver_image = self.resource_spec.solver.image
        self.solver_gpus = self.resource_spec.solver.gpus

    def _verify_backend(self) -> Optional[dict]:
        """Build the isolated verifier-container backend from the verifier slice.

        Returns None (host-subprocess verify, the default) UNLESS the task ships a
        runnable checker (``_checker_dir`` set) OR the verifier slice explicitly asks
        for a container (gpus/cpus/memory/image). Pure-python/LLM tasks with no checker
        keep host verify — fast, no docker, and creds still reach it. When active, the
        verifier image defaults to the solver image (same torch/CUDA base) if the
        verifier slice names none, and the timeout comes from ``verifier.timeout_sec``."""
        v = self.resource_spec.verifier
        wants_container = (not v.is_empty()) and (
            v.gpus is not None or v.cpus is not None or v.memory_mb is not None
            or v.image is not None)
        if not self._checker_dir and not wants_container:
            return None
        return {
            "image": v.image or self.resource_spec.solver.image,
            "gpus": v.gpus,
            "cpus": v.cpus,
            "memory_mb": v.memory_mb,
            "allow_internet": v.allow_internet,
            "timeout_s": v.timeout_sec,
        }

    def _llm_env(self) -> dict:
        """The host-side LLM cred env built from ``llm_config`` (present keys only).

        Maps {api_key,base_url,model} -> LLM_API_KEY/LLM_BASE_URL/LLM_MODEL. This is the
        ONLY dict pushed into eval_service._eval_env, and from there only into the
        host-side verifier/feedback subprocess — never the Solver container."""
        cfg = self.llm_config or {}
        env = {}
        if cfg.get("api_key"):
            env["LLM_API_KEY"] = str(cfg["api_key"])
        if cfg.get("base_url"):
            env["LLM_BASE_URL"] = str(cfg["base_url"])
        if cfg.get("model"):
            env["LLM_MODEL"] = str(cfg["model"])
        return env

    def _materialize_checker(self, ws: Path) -> None:
        """Copy an agent-bundled ``checker/`` out of the workspace to a stable host path.

        The bootstrap agent puts a wrapped oracle/scorer under ``<ws>/checker/``. We copy
        it once to ``run_dir/checker/`` (a stable location that survives workspace churn)
        and record its absolute path in ``_checker_dir`` for ``ctx["checker_dir"]``. No-op
        when the problem ships no checker."""
        src = ws / "checker"
        if not src.is_dir():
            self._checker_dir = None
            return
        dst = self.run_dir / "checker"
        if not dst.exists():
            _copy_tree(src, dst)
        self._checker_dir = str(dst.resolve())

    def _recover_bootstrap(self, ws: Path, note: str, stderr: str = "") -> Bootstrap:
        vf = ws / "verifier.py"
        if not vf.is_file():
            tail = (stderr or "").strip()[-400:]
            raise AgentSystemUnavailable(
                f"bootstrap agent did not author verifier.py (note: {note[:200]}"
                + (f"; stderr: {tail}" if tail else "") + ")")
        src = vf.read_text(encoding="utf-8")
        ctx = _read_json(ws / "ctx.json", default={})
        if not isinstance(ctx, dict):
            ctx = {}
        seed = _read_json(ws / "seed_solution.json", default={})
        probes_raw = _read_json(ws / "probes.json", default=[])
        probes = [p for p in probes_raw if isinstance(p, dict) and "solution" in p] \
            if isinstance(probes_raw, list) else []
        brief = ws / "SOLVER_BRIEF.md"
        notes = brief.read_text(encoding="utf-8")[:2000] if brief.is_file() else ""
        # Optional agent-authored feedback module + solver-env overrides.
        ff = ws / "feedback.py"
        feedback_src = ff.read_text(encoding="utf-8") if ff.is_file() else None
        solver_env = _read_json(ws / "solver_env.json", default={})
        if not isinstance(solver_env, dict):
            solver_env = {}
        # Cold admissibility policy — default to no-proof when the file is missing or
        # malformed (conservative: an optimization task must never inherit the proof
        # escape hatch just because the agent forgot to write the file).
        policy = _read_json(ws / "reframe_policy.json", default={})
        admits_proof = bool(policy.get("admits_proof", False)) \
            if isinstance(policy, dict) else False
        provable_claim = str(policy.get("provable_claim", "")) \
            if isinstance(policy, dict) else ""
        # If the agent bundled a checker, pin its host path into ctx BEFORE the smoke
        # test so a wrapping verifier can reach it exactly as it will at run time.
        self._materialize_checker(ws)
        if self._checker_dir:
            ctx = {**ctx, "checker_dir": self._checker_dir}
        # Resolve the resource spec now so the smoke test runs the SEED through the same
        # (possibly containerized/GPU) verifier backend it will use at run time — a
        # checker-wrapping V0 that shells to torch must be smoked in-container, not on
        # the torch-less host (else every candidate would be a false infeasible).
        self._resolve_resources()
        self._apply_solver_env_fallback(solver_env)
        backend = self._verify_backend()
        # smoke-test V0 on the seed before trusting it as the black box.
        probe_ev = Evaluator(exec_backend=backend)
        probe_ev.versions.append(_version0(src, feedback_src))
        r = probe_ev.run(seed, ctx, env=self._llm_env() or None)
        if r.error:
            raise AgentSystemUnavailable(
                f"bootstrap verifier.py fails on its own seed: {r.error[:200]}")
        # A clean run that still reports infeasible means the checker ran but rejected
        # the seed — surface it loudly instead of silently proceeding all-infeasible.
        if not r.feasible:
            self.store.event("bootstrap_verify_infeasible",
                             checker_dir=self._checker_dir, raw=r.raw,
                             backend=("container" if backend else "host"))
        return Bootstrap(verifier_src=src, ctx=ctx, seed_solution=seed,
                         probes=probes, notes=notes, feedback_src=feedback_src,
                         solver_env=solver_env, admits_proof=admits_proof,
                         provable_claim=provable_claim)

    # ================================================================
    # LOOP 1 — solve: long-lived Solver container + file-state resume
    # LOOP 2 — evolve: agent-authored detect + harden on review / cadence
    # ================================================================
    def solve_and_evolve(self, *, max_turns: int = 12) -> None:
        assert self.gateway and self.agent_elf and self.eval_service
        sol_ws = self.run_dir / "solver_ws"
        sol_ws.mkdir(parents=True, exist_ok=True)
        self._refresh_solver_ws(sol_ws)

        control = ControlSocket(workdir=sol_ws, handler=self._handle_shim)
        control.seed_shims(sol_ws)
        control.start()
        sol = self.resource_spec.solver
        # If the socket had to bind outside the workdir (AF_UNIX path too long),
        # hand the container the real bind path so it mounts it at /work/.control.sock.
        sock_mount = control.host_path if control.needs_explicit_mount else None
        container = DockerContainer(workdir=sol_ws, gateway=self.gateway,
                                    agent_elf=self.agent_elf,
                                    image=self.solver_image or self.image,
                                    gpus=self.solver_gpus,
                                    cpus=sol.cpus, memory_mb=sol.memory_mb,
                                    allow_internet=sol.allow_internet,
                                    control_sock_host_path=sock_mount)
        self.store.event("solver_container_start",
                         image=self.solver_image or self.image, gpus=self.solver_gpus,
                         cpus=sol.cpus, memory_mb=sol.memory_mb,
                         allow_internet=sol.allow_internet)
        try:
            container.start()
            turn = 0
            repush = False   # set when the LAST turn ended voluntarily before deadline
            while True:
                # A harden that moved V leaves us OWING one solve turn so the
                # Solver re-baselines under the new verifier — that owed turn runs
                # even past the nominal deadline. Otherwise stop on turns/deadline.
                owed = self._owe_post_harden_solve
                if not owed:
                    if turn >= max_turns or self.deadline.expired():
                        break
                turn += 1
                self._turn = turn
                # Normal turns leave a reserve so the LAST review before the
                # deadline can still harden AND get its post-harden solve. The owed
                # turn itself is not reserve-capped — it is the post-harden solve.
                remaining = self.deadline.remaining()
                if owed:
                    turn_budget = max(self.min_solver_turn_s,
                                      min(self.post_harden_solve_s, remaining)
                                      if remaining > 0 else self.post_harden_solve_s)
                else:
                    reserve = self.harden_timeout_s + self.post_harden_solve_s
                    turn_budget = remaining - reserve
                    if turn_budget < self.min_solver_turn_s:
                        # not enough left for a normal turn; only proceed if we can
                        # still fit the floor, else stop and let finalize run.
                        if remaining <= self.min_solver_turn_s:
                            break
                        turn_budget = self.min_solver_turn_s
                self.store.event("solver_turn_start", turn=turn,
                                 remaining_s=round(remaining, 1),
                                 turn_budget_s=round(turn_budget, 1),
                                 post_harden=owed)
                if owed:
                    self._owe_post_harden_solve = False
                    self._post_harden_solves += 1
                    # A modality switch changed the whole game — re-seed the solver
                    # workspace with the NEW brief/seed BEFORE the re-baseline turn.
                    if self._pending_mode_refresh:
                        self._refresh_solver_ws(sol_ws)
                        self._pending_mode_refresh = False
                session = container.exec_agent(
                    self._solver_prompt(turn, repush=(repush and not owed)),
                    timeout_s=turn_budget)
                repush = False   # consumed; recomputed below for the next turn
                self.store.cost(who="solver", kind="agent_turn", calls=1,
                                ok=session.ok, turn=turn)
                self._extract_best(sol_ws)

                # did the agent leave a review request? (its reason for exiting)
                review_req = sol_ws / "review_request.json"
                if review_req.is_file():
                    before = self.eval_service.current_version()  # type: ignore[union-attr]
                    self._handle_review_request(review_req, sol_ws)
                    review_req.unlink()
                    # if V moved, OWE a re-baseline solve turn (guaranteed to run).
                    if self.eval_service.current_version() > before:  # type: ignore[union-attr]
                        self._owe_post_harden_solve = True
                    continue   # resume: next turn in the SAME container

                # The turn we just ran WAS the guaranteed post-harden re-baseline; do
                # NOT harden again right after it. Loop to the top so a fresh normal
                # turn (if budget remains) drives the next round, else we stop there.
                # This bounds over-run to the single owed turn already in flight.
                if owed:
                    continue

                # A NORMAL turn finished — cleanly OR by hitting its wall-clock cap.
                # In BOTH cases the ORCHESTRATOR (not the agent) drives the evolve
                # step: red-team + maybe harden. This is "the system moves the second
                # Solver" the design calls for — we never depend on the agent
                # voluntarily exiting (codex often just works until it is stopped, as
                # the first Chowla/peak runs showed). A timed-out normal turn still
                # has the reserved harden+post-harden budget, so hardening is safe.
                if self._proactive_supervise(sol_ws):
                    self._owe_post_harden_solve = True
                    continue
                # V is stable after the Supervisor looked. Wall-clock is the ONLY stop
                # signal (SForge "timeout = done"): we do NOT stop just because the
                # agent voluntarily exited with a stable V. If it hit its wall clock it
                # still had work → give it another turn on the same V. If it stopped on
                # its own before the deadline, that stop is itself a signal — the
                # Supervisor just reviewed it and chose NOT to change the game, so the
                # solver was only slacking: kick it back in with an added push. Either
                # way, keep going until the deadline (loop-top) or max_turns stops us.
                if not session.note.startswith("agent hit the wall-clock"):
                    repush = True
                continue
        finally:
            container.stop()
            control.stop()
            self.store.event("solver_container_stop")
        self._finalize()

    # -- shim handler (synchronous socket channel) ------------------------
    def _handle_shim(self, cmd: str, arg: dict) -> dict:
        svc = self.eval_service
        assert svc is not None
        if cmd == "eval":
            r = svc.query(arg.get("solution", {}) or {})
            self.store.cost(who="solver", kind="eval_query", calls=1)
            if r.ok and r.score is not None and r.score > self._best_score:
                self._best_score = r.score
                self._best_payload = arg.get("solution", {})
            return {"ok": r.ok, "score": r.score, "feasible": r.feasible,
                    "artifacts": r.artifacts, "verifier_version": r.verifier_version,
                    "feedback_level": r.feedback_level, "detail": r.detail}
        if cmd == "status":
            return {"verifier_version": svc.current_version(),
                    "feedback_level": svc.feedback_level.value,
                    "deadline_remaining_s": round(self.deadline.remaining(), 1)}
        if cmd == "ask":
            # the agent asks synchronously but the heavy path is file-state resume;
            # acknowledge and tell it to write review_request.json + exit.
            return {"ok": True,
                    "instruction": ("write your best solution and your question to "
                                    "review_request.json (as {\"solution\":...,"
                                    "\"question\":...}) and EXIT; the supervisor will "
                                    "review and you will resume with a fresh verifier_version")}
        return {"ok": False, "error": f"unknown cmd {cmd!r}"}

    # -- the evolve loop: agent-authored review + harden ------------------
    def _handle_review_request(self, req_path: Path, sol_ws: Path) -> None:
        try:
            req = json.loads(req_path.read_text())
        except (json.JSONDecodeError, OSError):
            req = {}
        best = req.get("solution") or self._best_payload or self._seed
        self.reviews_handled += 1
        self.store.event("review_request", question=str(req.get("question", ""))[:300])
        self._run_supervisor_harden(best, sol_ws, trigger="review_request")

    def _proactive_supervise(self, sol_ws: Path) -> bool:
        """Orchestrator-driven cadence: red-team + maybe harden. Returns True if V moved."""
        best = self._best_payload or self._seed
        before = self.eval_service.current_version()   # type: ignore[union-attr]
        self._run_supervisor_harden(best, sol_ws, trigger="proactive")
        return self.eval_service.current_version() > before   # type: ignore[union-attr]

    def _run_supervisor_harden(self, best: dict, sol_ws: Path, *,
                               trigger: str) -> None:
        # Control-arm switch: keep the bootstrap V0 verifier frozen and never let the
        # Supervisor red-team / harden / evolve it. This is the ONLY harden entry point
        # (both _handle_review_request and _proactive_supervise funnel here), so a single
        # early-return disables all evolution while leaving bootstrap, the solver loop,
        # _extract_best and _finalize untouched — the Solver just mines the frozen V0 for
        # the full wall-clock budget. verifier stays v0, hardenings stays 0.
        if self.freeze_verifier:
            self.store.event("harden_skipped", trigger=trigger, reason="freeze_verifier")
            return
        assert self.gateway and self.agent_elf and self.eval_service
        ws = self.run_dir / "harden_ws"
        if ws.exists():
            _rmtree(ws)
        ws.mkdir(parents=True, exist_ok=True)
        _copy_tree(self.raw_input_dir, ws / "problem")
        human_guidance = self.run_dir / "human_guidance.md"
        if human_guidance.is_file():
            (ws / "HUMAN_GUIDANCE.md").write_text(
                human_guidance.read_text(encoding="utf-8"), encoding="utf-8"
            )
        (ws / "current_verifier.py").write_text(
            self.eval_service.current_source(), encoding="utf-8")
        cur_fb = self.eval_service.current_feedback_source()
        if cur_fb is not None:
            (ws / "current_feedback.py").write_text(cur_fb, encoding="utf-8")
        (ws / "best_solution.json").write_text(json.dumps(best, indent=2))
        (ws / "seed_solution.json").write_text(json.dumps(self._seed, indent=2))
        (ws / "probe_report.json").write_text(
            json.dumps(self._probe_report(), indent=2))
        # Solver's own log + review request are FILE channels into the Supervisor's
        # /work. The solver records concrete execution observations here (e.g. a build
        # cache / permission / toolchain failure that makes EVERY candidate score 0 for
        # reasons unrelated to the verifier's judgment). Without this the Supervisor only
        # sees score=0 and misreads a harness/infra fault as a verifier weakness, then
        # hardens forever without touching the real cause. Copy whatever the solver left.
        for _name in ("scratchpad.md", "review_request.json"):
            _src = sol_ws / _name
            if _src.is_file():
                try:
                    (ws / f"solver_{_name}").write_text(
                        _src.read_text(encoding="utf-8", errors="replace"),
                        encoding="utf-8")
                except OSError:
                    pass
        remaining = self.deadline.remaining()
        # progress.json is CONTEXT for the Supervisor's own judgment, NOT a "plateau"
        # verdict the orchestrator computes: the score trajectory, how long the current
        # game has been mined (wall-clock since the last mode switch), time left, mode,
        # and turn. The Supervisor decides for itself whether to bite harder or reframe.
        (ws / "progress.json").write_text(json.dumps({
            "mode": self._mode,
            "turn": self._turn,
            "recent_scores": self._score_trajectory[-8:],
            "spent_in_mode_s": round(self.deadline.elapsed() - self._mode_since_s, 1),
            "remaining_s": round(remaining, 1),
        }, indent=2))

        # Harden runs on its OWN protected timeout, not `remaining`: the loop keeps
        # (harden_timeout_s + post_harden_solve_s) in reserve, so even a review that
        # fires right at the deadline still gets a full harden. Capping by `remaining`
        # here is what starved the harden to ~0s in the first Chowla run.
        self.store.event("harden_start", trigger=trigger, mode=self._mode,
                         remaining_s=round(remaining, 1))
        # Inject the SWITCH/REFRAME option (c) ONLY for problems cold-judged to admit a
        # proof/disproof. Optimization/engineering tasks get the slim menu (tighten +
        # expose-more) and can never escape a plateau into a subjective proof game.
        if self._admits_proof:
            claim = self._provable_claim or "(the admissible claim recorded at bootstrap)"
            prompt = _HARDEN_PROMPT.format(
                reframe_option=_REFRAME_OPTION_ALLOWED.format(provable_claim=claim),
                reframe_bias=_REFRAME_BIAS_ALLOWED,
                reframe_output=_REFRAME_OUTPUT_ALLOWED,
            )
        else:
            prompt = _HARDEN_PROMPT.format(
                reframe_option=_REFRAME_OPTION_WITHHELD,
                reframe_bias=_REFRAME_BIAS_WITHHELD,
                reframe_output=_REFRAME_OUTPUT_WITHHELD,
            )
        self.store.event("harden_prompt_built", trigger=trigger,
                         admits_proof=self._admits_proof)
        session = one_shot_agent(
            ws, self.gateway, self.agent_elf, prompt,
            timeout_s=self.harden_timeout_s, image=self.image)
        self.store.cost(who="supervisor", kind="harden_session", calls=1, ok=session.ok)

        verdict = _read_json(ws / "verdict.json", default={})
        self._apply_harden(ws, trigger=trigger, verdict=verdict)
        human_guidance = self.run_dir / "human_guidance.md"
        if human_guidance.is_file():
            (sol_ws / "HUMAN_GUIDANCE.md").write_text(
                human_guidance.read_text(encoding="utf-8"), encoding="utf-8"
            )

    def _apply_harden(self, ws: Path, *, trigger: str, verdict: dict) -> None:
        """Install whatever the smith authored: a new verifier and/or a feedback module,
        possibly a full modality switch, after mechanical safety and expert review."""
        svc = self.eval_service
        assert svc is not None
        new_vf = ws / "verifier.py"
        new_ff = ws / "feedback.py"
        switch = _read_json(ws / "mode_switch.json", default={})
        is_switch = bool(isinstance(switch, dict) and switch.get("switch"))

        has_new_verifier = new_vf.is_file()
        has_new_feedback = new_ff.is_file()
        if not has_new_verifier and not has_new_feedback:
            self.store.review(trigger=trigger, gaming=bool(verdict.get("gaming")),
                              installed=False, mode=self._mode,
                              reasoning=str(verdict.get("reasoning", ""))[:400])
            return

        # The evaluation the smith is proposing. A verifier rewrite is authoritative;
        # absent, keep the current verifier and only swap/extend feedback.
        verify_src = (new_vf.read_text(encoding="utf-8") if has_new_verifier
                      else svc.current_source())
        feedback_src = (new_ff.read_text(encoding="utf-8") if has_new_feedback
                        else (None if is_switch else svc.current_feedback_source()))

        # On a modality switch the whole game changes representation, so validate under
        # the NEW seed/probes/ctx the agent authored; else validate under the current.
        if is_switch:
            new_seed = _read_json(ws / "seed_solution.json", default=self._seed)
            new_probes_raw = _read_json(ws / "probes.json", default=self._probes)
            new_probes = [p for p in new_probes_raw
                          if isinstance(p, dict) and "solution" in p] \
                if isinstance(new_probes_raw, list) else []
            new_ctx = _read_json(ws / "ctx.json", default=self._ctx)
            if not isinstance(new_ctx, dict):
                new_ctx = self._ctx
            if self._checker_dir:
                new_ctx = {**new_ctx, "checker_dir": self._checker_dir}
            val_seed, val_probes, val_ctx = new_seed, new_probes, new_ctx
        else:
            val_seed, val_probes, val_ctx = self._seed, self._probes, self._ctx

        err = svc.validate_evaluation(
            verify_src=verify_src, feedback_src=feedback_src,
            seed=val_seed, probes=val_probes, ctx=val_ctx)
        if err is not None:
            self.store.event("harden_rejected", reason=err[:200],
                             attempted_switch=is_switch)
            self.store.review(trigger=trigger, gaming=bool(verdict.get("gaming")),
                              installed=False, mode=self._mode,
                              reasoning=str(verdict.get("reasoning", ""))[:400])
            return

        # The proposal is valid. Human/Proxy consultation is an explicit Supervisor
        # decision, not an automatic gate on every harden. Routine, well-understood
        # anti-gaming fixes can install autonomously; ambiguous/high-impact changes
        # request review by setting verdict.json["ask_human"] = true. An absent field
        # is false so old Supervisor outputs remain autonomous rather than silently
        # consuming scarce Human Proxy sessions.
        # Require a JSON boolean, not a truthy string such as "false".
        ask_human = verdict.get("ask_human") is True
        self.store.event(
            "supervisor_human_decision",
            requested=ask_human,
            reason="verdict.ask_human" if ask_human else "not_requested",
        )
        if self.human_port is not None and ask_human:
            current_source = svc.current_source()
            proposal_diff = "".join(
                difflib.unified_diff(
                    current_source.splitlines(keepends=True),
                    verify_src.splitlines(keepends=True),
                    fromfile=f"v{svc.current_version()}.py",
                    tofile="proposed_verifier.py",
                )
            )
            human_outcome = self._consult_human(
                purpose="verifier_change",
                context={
                    "trigger": trigger,
                    "current_version": svc.current_version(),
                    "current_verifier": current_source,
                    "proposed_verifier": verify_src,
                    "diff": proposal_diff,
                    "proposed_feedback": feedback_src,
                    "verdict": verdict,
                    "is_mode_switch": is_switch,
                    "proposed_mode": switch.get("to_mode") if is_switch else self._mode,
                },
            )
            if human_outcome is None or human_outcome.decision.lower() != "approve":
                human_decision = (
                    human_outcome.decision if human_outcome is not None else "unavailable"
                )
                self.store.event(
                    "harden_held_for_human",
                    decision=human_decision,
                    trigger=trigger,
                )
                self.store.review(
                    trigger=trigger,
                    gaming=bool(verdict.get("gaming")),
                    installed=False,
                    mode=self._mode,
                    human_decision=human_decision,
                    reasoning=str(verdict.get("reasoning", ""))[:400],
                )
                return

        ver = svc.install_verifier(
            verify_src, origin="agent", feedback_src=feedback_src,
            note=f"harden ({trigger}): {str(verdict.get('reasoning',''))[:120]}")
        self.store.verifier_version(
            ver, verify_src, origin="agent",
            note=f"harden ({trigger}){' [mode_switch]' if is_switch else ''}",
            rationale=str(verdict.get("reasoning", ""))[:400],
            feedback_src=feedback_src)
        self.hardenings += 1

        if is_switch:
            # Swap the in-memory game AND overwrite the single-source-of-truth
            # bootstrap_ws so both the solve loop and a resume read the NEW contract.
            self._ctx, self._seed, self._probes = val_ctx, val_seed, val_probes
            self._mode = str(switch.get("to_mode", "proof"))
            self._score_trajectory = []   # a new game resets the score window
            self._mode_since_s = self.deadline.elapsed()  # current game starts now
            self._pending_mode_refresh = True
            self._overwrite_bootstrap_contract(ws)
            self.store.event("mode_switch", to_mode=self._mode,
                             reasoning=str(switch.get("reasoning", ""))[:300])
        self.store.review(trigger=trigger, gaming=bool(verdict.get("gaming")),
                          installed=True, mode=self._mode, mode_switch=is_switch,
                          reasoning=str(verdict.get("reasoning", ""))[:400])

    def _overwrite_bootstrap_contract(self, ws: Path) -> None:
        """Persist a switched game's new contract into bootstrap_ws (the resume source).

        The solve loop and ``_resume_from_disk`` both read seed/probes/ctx/brief from
        ``bootstrap_ws``; after a modality switch that must reflect the NEW game."""
        bws = self.run_dir / "bootstrap_ws"
        bws.mkdir(parents=True, exist_ok=True)
        (bws / "ctx.json").write_text(json.dumps(self._ctx, indent=2), encoding="utf-8")
        (bws / "seed_solution.json").write_text(
            json.dumps(self._seed, indent=2), encoding="utf-8")
        (bws / "probes.json").write_text(
            json.dumps(self._probes, indent=2), encoding="utf-8")
        new_brief = ws / "SOLVER_BRIEF.md"
        if new_brief.is_file():
            (bws / "SOLVER_BRIEF.md").write_text(
                new_brief.read_text(encoding="utf-8"), encoding="utf-8")

    def _probe_report(self) -> list:
        """Score each bootstrap probe under the current V — evidence for the smith."""
        out = []
        svc = self.eval_service
        for p in self._probes:
            sol = p.get("solution", {})
            r = svc.query(sol) if svc else None   # type: ignore[union-attr]
            out.append({"description": p.get("description", ""),
                        "score": (r.score if r else None),
                        "feasible": (r.feasible if r else None)})
        return out

    # -- solver workspace + prompts ---------------------------------------
    def _refresh_solver_ws(self, ws: Path) -> None:
        """(Re)seed the solver workspace from the CURRENT contract in bootstrap_ws.

        Called at start and again right after a modality switch (the switch overwrote
        bootstrap_ws with the new game's brief/seed). The scratchpad is preserved so
        the solver keeps its own memory across the switch."""
        brief = self.run_dir / "bootstrap_ws" / "SOLVER_BRIEF.md"
        if brief.is_file():
            (ws / "PROBLEM_BRIEF.md").write_text(brief.read_text(encoding="utf-8"),
                                                 encoding="utf-8")
        human_guidance = self.run_dir / "human_guidance.md"
        if human_guidance.is_file():
            (ws / "HUMAN_GUIDANCE.md").write_text(
                human_guidance.read_text(encoding="utf-8"), encoding="utf-8"
            )
        (ws / "seed_solution.json").write_text(json.dumps(self._seed, indent=2))
        (ws / "solution_out.json").write_text(json.dumps(self._seed))
        if not (ws / "scratchpad.md").exists():
            (ws / "scratchpad.md").write_text(
                "# Solver scratchpad\n\nRecord findings here so future turns resume "
                "fast. This file persists across turns.\n", encoding="utf-8")

    def _solver_prompt(self, turn: int, *, repush: bool = False) -> str:
        mode_line = ""
        human_line = ""
        if (self.run_dir / "human_guidance.md").is_file():
            human_line = (
                " Read HUMAN_GUIDANCE.md too: it contains authoritative expert "
                "clarifications and evidence requests from completed Human Sessions."
            )
        if self._mode != "construction":
            mode_line = (
                f"\n\nNOTE: the evaluation's GAME has been REFRAMED — it is now "
                f"'{self._mode}' mode. Your solution_out.json payload is no longer a "
                "bigger construction but the ARGUMENT the reframed PROBLEM_BRIEF.md now "
                "describes (e.g. a proof, or a disproof/counterexample), scored "
                "qualitatively by a hidden LLM-verifier that returns natural-language "
                "guidance in the eval response `detail`. Re-read PROBLEM_BRIEF.md, then "
                "read `detail` each eval and follow it — a higher score reflects genuine "
                "progress in this game, not a bigger construction.")
        repush_line = ""
        if repush:
            repush_line = (
                "\n\nKEEP PUSHING: the run is NOT over — wall-clock remains (check "
                "./container-status). You exited last turn with the hidden verifier "
                "UNCHANGED: the Supervisor already reviewed your stop and deliberately "
                "did NOT change the game, which means you were not actually stuck — you "
                "stopped early. Do not idle or declare victory. Attack the SAME target "
                "again from a genuinely different angle (a new algorithmic idea, a "
                "sharper construction, an assumption you have not tried relaxing) and "
                "push the score higher. Only if you are truly blocked — not merely done "
                "— run ./container-ask-supervisor solution_out.json to escalate; do not "
                "just exit again.")
        return (
            f"You are the SOLVER (turn {turn}). Read PROBLEM_BRIEF.md and scratchpad.md "
            "in this directory (/work) first — scratchpad.md is your own memory from "
            f"previous turns, keep appending to it.{human_line}\n\n"
            "Score any candidate solution with:  ./container-eval <solution.json>  "
            "(higher score is better; it returns verifier_version too). Check remaining "
            "time with:  ./container-status .\n\n"
            "Watch verifier_version in each eval response: if it INCREASED since last "
            "turn, the Supervisor hardened the hidden verifier — your old best may be "
            "stale, so re-baseline. Do NOT merely game the verifier; a Supervisor is "
            "red-teaming it.\n\n"
            "Write your single best solution as JSON to solution_out.json (overwrite "
            "when you improve it). If you plateau or suspect your best is exploiting a "
            "verifier flaw, run  ./container-ask-supervisor solution_out.json , then "
            "follow its instruction (write review_request.json and exit) so the "
            "Supervisor can review and you resume next turn with a fresh verifier. "
            "Otherwise, keep improving until you are confident, then stop."
            + mode_line + repush_line
        )

    def _extract_best(self, ws: Path) -> None:
        out = ws / "solution_out.json"
        if not out.is_file():
            return
        try:
            payload = json.loads(out.read_text())
        except (json.JSONDecodeError, OSError):
            return
        r = self.eval_service.query(payload)   # type: ignore[union-attr]
        self.store.candidate(payload, {"score": r.score,
                                       "verifier_version": r.verifier_version},
                             score=r.score)
        if r.ok and r.score is not None and r.score > self._best_score:
            self._best_score, self._best_payload = r.score, payload
        # Feed the score trajectory: the best-so-far after this extract (finite only).
        if self._best_score != float("-inf"):
            self._score_trajectory.append(self._best_score)
        self.store.trajectory(event="best_extracted", best_score=self._best_score,
                              verifier_version=r.verifier_version, mode=self._mode)

    # -- finalize ---------------------------------------------------------
    def _finalize(self) -> None:
        svc = self.eval_service
        final_v = svc.current_version() if svc else -1
        self.store.event("run_stop", final_verifier_version=final_v,
                         hardenings=self.hardenings, reviews=self.reviews_handled,
                         post_harden_solves=self._post_harden_solves,
                         best_score=self._best_score)
        self.store.finalize_manifest({
            "final_verifier_version": final_v,
            "verifier_hardenings": self.hardenings,
            "reviews_handled": self.reviews_handled,
            "post_harden_solves": self._post_harden_solves,
            "final_mode": self._mode,
            "best_score": self._best_score if self._best_score != float("-inf") else None,
            "best_solution": self._best_payload,
            "resource_spec": self.resource_spec.to_manifest(),
        })

    # -- resume -----------------------------------------------------------
    def can_resume(self) -> bool:
        """True if this run_dir already holds a completed bootstrap to resume from.

        The 8h runs must survive a crash/OOM/host reboot. Everything needed to
        continue is already on disk (the store never depended on in-memory state):
        the authored evaluator chain (``verifier_versions/v*.py``), the instance
        (``bootstrap_ws/{ctx,seed_solution,probes}.json``), and the best-so-far
        (``manifest.json`` / ``solver/candidates/``). ``can_resume`` just checks the
        bootstrap boundary was crossed."""
        bws = self.run_dir / "bootstrap_ws"
        return ((bws / "verifier.py").is_file()
                and (self.run_dir / "supervisor" / "verifier_versions" / "v0.py").is_file())

    def _resume_from_disk(self) -> None:
        """Rebuild evaluator + counters + best-so-far from ``runs/<id>/`` (no re-bootstrap)."""
        vdir = self.run_dir / "supervisor" / "verifier_versions"
        bws = self.run_dir / "bootstrap_ws"
        # 1. rebuild the FULL version chain in order (v0 = bootstrap, v1.. = hardens).
        metas = {}
        for line in _read_lines(vdir.parent / "versions.jsonl"):
            try:
                m = json.loads(line)
                metas[int(m.get("version", -1))] = m
            except (json.JSONDecodeError, ValueError):
                continue
        from ..demo.evaluator import VerifierVersion
        self.evaluator = Evaluator()
        n = 0
        while (vdir / f"v{n}.py").is_file():
            src = (vdir / f"v{n}.py").read_text(encoding="utf-8")
            m = metas.get(n, {})
            # A versioned feedback module (free-form disclosure) is a sibling written
            # only when that version had one — rebuild the exact (verify, feedback) pair.
            ff = vdir / f"v{n}.feedback.py"
            fb_src = ff.read_text(encoding="utf-8") if ff.is_file() else None
            self.evaluator.versions.append(
                VerifierVersion(n, src, m.get("origin", "agent"), m.get("note", ""),
                                feedback_src=fb_src))
            n += 1
        if not self.evaluator.versions:   # defensive: fall back to the bootstrap source
            self.evaluator.versions.append(_version0((bws / "verifier.py").read_text()))
        # each version past v0 is one installed hardening.
        self.hardenings = max(0, len(self.evaluator.versions) - 1)
        # 2. instance + red-team, from the bootstrap workspace (already overwritten to
        #    the CURRENT game if a modality switch happened).
        self._ctx = _read_json(bws / "ctx.json", default={})
        if not isinstance(self._ctx, dict):
            self._ctx = {}
        self._seed = _read_json(bws / "seed_solution.json", default={})
        probes_raw = _read_json(bws / "probes.json", default=[])
        self._probes = [p for p in probes_raw
                        if isinstance(p, dict) and "solution" in p] \
            if isinstance(probes_raw, list) else []
        # Re-establish a bundled checker path (survives under run_dir/checker/) and the
        # solver-env overrides, then re-pin ctx["checker_dir"] to the current host path.
        ckdir = self.run_dir / "checker"
        if ckdir.is_dir():
            self._checker_dir = str(ckdir.resolve())
            self._ctx = {**self._ctx, "checker_dir": self._checker_dir}
        solver_env = _read_json(bws / "solver_env.json", default={})
        self._resolve_resources()
        self._apply_solver_env_fallback(solver_env if isinstance(solver_env, dict) else {})
        # Restore the cold admissibility policy (default no-proof if absent — old runs
        # predating reframe_policy.json stay in construction, which is the safe game).
        policy = _read_json(bws / "reframe_policy.json", default={})
        self._admits_proof = bool(policy.get("admits_proof", False)) \
            if isinstance(policy, dict) else False
        self._provable_claim = str(policy.get("provable_claim", "")) \
            if isinstance(policy, dict) else ""
        # 3. reviews handled = count of review_request events already processed; the
        #    current GAME mode = the last mode_switch recorded (else construction).
        self.reviews_handled = sum(
            1 for ln in _read_lines(self.run_dir / "events.jsonl")
            if '"review_request"' in ln)
        self._mode = self._recover_mode_from_events()
        # 4. wire the black box, then recover best-so-far by re-scoring under CURRENT V.
        self.eval_service = EvalService(
            evaluator=self.evaluator, ctx_provider=lambda: self._ctx,
            feedback_level=self.feedback_level, _eval_env=dict(self._llm_env()),
            _verify_backend=self._verify_backend())
        self.eval_service.on_query = lambda rec, res: self.store.query(_query_line(rec))
        self._recover_best_from_disk()
        self.store.event("resume", from_version=self.eval_service.current_version(),
                         hardenings=self.hardenings, reviews=self.reviews_handled,
                         mode=self._mode,
                         best_score=(self._best_score
                                     if self._best_score != float("-inf") else None))

    def _recover_mode_from_events(self) -> str:
        """The current game mode = the ``to_mode`` of the last ``mode_switch`` event."""
        mode = "construction"
        for ln in _read_lines(self.run_dir / "events.jsonl"):
            if '"mode_switch"' not in ln:
                continue
            try:
                ev = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if ev.get("kind") == "mode_switch" and ev.get("to_mode"):
                mode = str(ev["to_mode"])
        return mode

    def _recover_best_from_disk(self) -> None:
        """Re-score prior candidates + the manifest best under the CURRENT verifier.

        Scores from earlier versions are stale after a harden, so we recompute rather
        than trust the recorded number — the best under v0 may be infeasible under v2."""
        svc = self.eval_service
        assert svc is not None
        cands: list[dict] = []
        cdir = self.run_dir / "solver" / "candidates"
        if cdir.is_dir():
            for cf in sorted(cdir.glob("cand_*.json")):
                obj = _read_json(cf, default={})
                if isinstance(obj, dict) and isinstance(obj.get("payload"), dict):
                    cands.append(obj["payload"])
        man = _read_json(self.run_dir / "manifest.json", default={})
        if isinstance(man.get("best_solution"), dict):
            cands.append(man["best_solution"])
        for payload in cands:
            r = svc.query(payload)
            if r.ok and r.score is not None and r.score > self._best_score:
                self._best_score, self._best_payload = r.score, payload

    # -- top-level entry --------------------------------------------------
    def run(self, *, max_turns: int = 12, resume: bool = False) -> "AgentSystem":
        self.preflight()
        if resume and self.can_resume():
            self.store.event("run_start", budget_s=self.budget_s, mode="agent_system",
                             resumed=True)
            self._consult_human(
                purpose="task_definition",
                context={"phase": "resume_before_solving"},
                checkpoint=True,
            )
            self._resume_from_disk()
            self.solve_and_evolve(max_turns=max_turns)
            return self
        self.store.write_manifest({
            "mode": "agent_system",
            "raw_input_dir": str(self.raw_input_dir),
            "budget_s": self.budget_s,
            "image": self.image,
            "initial_feedback_level": self.feedback_level.value,
        })
        self.store.event("run_start", budget_s=self.budget_s, mode="agent_system")
        self._consult_human(
            purpose="task_definition",
            context={"phase": "before_bootstrap"},
            checkpoint=True,
        )
        self.bootstrap()
        self.solve_and_evolve(max_turns=max_turns)
        return self

    def summary(self) -> dict:
        return {
            "final_verifier_version":
                self.eval_service.current_version() if self.eval_service else -1,
            "verifier_hardenings": self.hardenings,
            "reviews_handled": self.reviews_handled,
            "post_harden_solves": self._post_harden_solves,
            "final_mode": self._mode,
            "best_score": self._best_score if self._best_score != float("-inf") else None,
            "best_solution": self._best_payload,
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _version0(src: str, feedback_src: Optional[str] = None):
    from ..demo.evaluator import VerifierVersion
    return VerifierVersion(0, src, "agent", "supervisor bootstrap",
                           feedback_src=feedback_src)


def _read_json(path: Path, *, default):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _read_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []


def _copy_tree(src: Path, dst: Path) -> None:
    import shutil as _sh
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        _sh.copy2(src, dst / src.name)
        return
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            _sh.copy2(item, target)


def _rmtree(path: Path) -> None:
    import shutil as _sh
    _sh.rmtree(path, ignore_errors=True)


def _query_line(rec) -> dict:
    return {
        "query_id": rec.query_id, "t": rec.t, "who": rec.who,
        "verifier_version": rec.verifier_version, "feedback_level": rec.feedback_level,
        "payload_summary": rec.payload_summary, "returned_score": rec.returned_score,
        "returned_keys": rec.returned_keys,
    }
