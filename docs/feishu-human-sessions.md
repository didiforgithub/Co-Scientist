# Feishu Human Sessions

Co-Scientist can pause at high-value checkpoints and open a direct-message conversation with one human expert in Feishu. The scarce unit is a **session**, not a message: each Co run may initiate at most five sessions, and every session may contain unlimited natural-language turns until the expert explicitly confirms closure.

## Human experience

The expert does not need commands or a special UI.

1. The bot sends a direct message explaining what Co-Scientist wants to discuss and shows the run's session usage, for example `第 1/5 次`.
2. The expert talks normally. Every message is delivered to a session-scoped Supervisor agent with the complete conversation and an allowlisted snapshot of current run evidence.
3. The agent can explain the task contract, candidate trajectory, evaluation queries, Supervisor reviews, verifier source/diff, and red-team probes. If evidence is absent it must say so rather than inventing it.
4. The expert says something natural such as `这轮结束吧` when they are done.
5. The agent summarizes conclusions and unresolved questions and asks for an explicit `确认结束`.
6. Only that second confirmation closes the session. `先别结束，我还有问题` returns it to normal conversation without consuming another session.

Session states are `OPEN -> ACTIVE -> CLOSE_REQUESTED -> CLOSED`, with `PAUSED` used for an interrupted listener. Idle time never closes a session and never consumes a new one.

## Feishu application configuration

Open the enterprise self-built application in [Feishu Developer Console](https://open.feishu.cn/app).

Required bot scopes:

- `im:message:send_as_bot`
- `im:message.p2p_msg:readonly`
- `im:message:readonly`

Recommended when group mentions or attachments are enabled later:

- `im:message.group_at_msg:readonly`
- `im:chat:read`
- `im:resource`

Application settings:

1. Add the **Bot** application capability.
2. Under **Events and Callbacks**, select long-connection event delivery.
3. Subscribe to `im.message.receive_v1`.
4. Include the expert in the application's availability scope.
5. Create and publish an application version after scopes/events change.

The runtime uses bot identity for both message calls and event consumption. The expert identifier must be an Feishu `open_id` beginning with `ou_`.

## Running a Co with Human Sessions

The machine that runs Co-Scientist must also have a configured `lark-cli`, the Codex agent binary/auth used by the existing AgentSystem, and Docker.

```bash
python -m coscientist.coevo.cli \
  --input /path/to/problem \
  --runs-dir /path/to/runs \
  --run-id abc-human-001 \
  --budget-hours 48 \
  --solver-strength strong \
  --feishu-expert-id ou_xxxxxxxxx \
  --lark-cli-executable lark-cli \
  --human-agent-model gpt-5.6-luna \
  --human-agent-reasoning-effort low \
  --human-agent-timeout-s 180
```

Human Session agents default to `gpt-5.6-luna` with low reasoning effort so a
human reply does not inherit the slower model/effort selected for the main
scientific run. The flags configure the Co-side evidence agent and, in proxy mode,
the independent Human Proxy; they do not change the Solver or Supervisor.

Resume the same run after a process or listener interruption:

```bash
python -m coscientist.coevo.cli \
  --input /path/to/problem \
  --runs-dir /path/to/runs \
  --run-id abc-human-001 \
  --resume \
  --feishu-expert-id ou_xxxxxxxxx
```

The task-definition checkpoint runs before bootstrap so its guidance is available while V0 is authored. Later, every valid proposed verifier change starts a Human Session before installation. The evaluator remains on its previous version for the entire conversation:

- final decision `approve`: install the validated proposal after the session closes;
- `reject`, `guide`, or `none`: hold the proposal and preserve the human guidance for the next Supervisor/Solver turn;
- five-session budget exhausted: do not initiate a sixth conversation and hold any evaluator change that lacks an explicit approval.

Disabling `--feishu-expert-id` preserves the existing autonomous behavior exactly.

## Model-backed Human Proxy Agent

For automated experiments and pre-Feishu testing, replace the Feishu expert with
a separate model agent and give it a private textual description of the real
evaluator:

```bash
python -m coscientist.coevo.cli \
  --input /path/to/problem \
  --runs-dir /path/to/runs \
  --run-id abc-proxy-001 \
  --solver-strength strong \
  --human-proxy-context /control-plane/evaluator_context.md \
  --human-agent-model gpt-5.6-luna \
  --human-agent-reasoning-effort low
```

The context should explain the evaluator's real goal, task semantics, important
failure modes, evidence standards, and useful environment constraints. It is
treated only as text: Co-Scientist never imports it as Python and never exposes an
evaluator callable or endpoint. Keep this private file outside both the run
directory and the Solver workspace.

The Human Proxy intentionally has the same information boundary as a remote human
expert:

| Participant | Receives | Does not receive |
|---|---|---|
| Co-side evidence agent | allowlisted run evidence and the durable public transcript | private Human Proxy evaluator context |
| Human Proxy agent | private evaluator text, session purpose/state, and the durable public transcript | direct solution payloads, internal session context, Solver/evaluator workspace, GPU, evaluator callable, or a provisioned evaluator endpoint |

Each Human Proxy turn runs in a fresh throw-away workspace containing only
`evaluator_context.md` and `transcript.json`. The workspace is deleted after
`turn.json` is read. The Proxy therefore cannot directly run the current
solution against the real evaluator. Its job is instead to act like a thoughtful
human expert: question assumptions, expose evaluator blind spots, and jointly
design evaluators, probes, feedback, and runtime environments that give the Solver
more truthful and actionable signals. Claims without run evidence must be labeled
as reasoning or advice.

The Proxy implements the same blocking `HumanInteractionPort` as Feishu and
reuses the same `HumanSessionStore` and `FeishuHumanSessionService` through an
in-memory loopback transport:

1. The Co-side evidence agent opens with what it knows from the current run.
2. The Human Proxy reads the private evaluator context and the public transcript,
   then asks or answers naturally.
3. Both agents may continue for any number of messages. The Proxy may request
   closure, reject an inaccurate close summary, and keep talking without opening a
   new session.
4. When the Proxy requests closure, the Co-side agent summarizes the conversation.
5. Only a later explicit confirmation closes the session and exposes the Proxy's
   reasoned structured outcome to `AgentSystem`.

The same five-session budget, unlimited turns, durable transcript, deduplication,
staged outcome, explicit closure confirmation, guidance persistence, and crash
semantics apply. `--feishu-expert-id` and `--human-proxy-context` are mutually
exclusive.

The driver has a 64-turn watchdog for each individual `consult` call. If a model
turn fails or that allowance is reached, the live session is paused. A later
`consult` resumes the same session without spending another slot and receives a
fresh watchdog allowance. This is a liveness guard, not a total message limit.
Once a final outcome is persisted in `proxy_state.json`, crash recovery reuses
that exact outcome rather than accepting a different judgment from a restarted
model turn.

## Persistence and crash recovery

All durable state lives inside the run directory:

```text
runs/<run_id>/
  human/
    index.json                         # immutable max=5 and opened count
    guidance.jsonl                    # structured outcomes across sessions
    checkpoints/task_definition.json  # prevents spending again on resume
    session_001/
      session.json                    # expert, purpose, context, lifecycle
      transcript.jsonl                # append-only human/agent/system turns
      pending_outcome.json            # close summary awaiting confirmation
      outcome.json                    # final confirmed structured outcome
      proxy_state.json                # final Proxy outcome frozen for safe resume
      outbox/*.json                    # generated replies awaiting Feishu delivery
      agent_workspace/
        context.json                  # bounded, redacted evidence copy
        transcript.json               # conversation copy
        response.json                 # agent's only writable result
  human_guidance.md                   # injected into Supervisor and Solver workspaces
```

Important recovery guarantees:

- Opening an already-live session for the same expert returns that session and does not spend another slot.
- Feishu `event_id` deduplicates inbound retries.
- Every outbound reply is persisted before sending and uses a stable Feishu idempotency key. A failed send remains in `outbox/` and is retried after restart.
- A staged outcome is not authoritative until the expert's separate close confirmation.
- A completed task-definition checkpoint is replayed from disk rather than reopened on `--resume`.
- Agent evidence is copied from an allowlist, bounded in size, and secret-shaped keys/values are redacted. The session agent never mounts the live solver or evaluator workspace.

For the direct-message MVP, run at most one Feishu-enabled Co at a time for the same expert and bot. Multiple simultaneous runs need a shared dispatcher to prevent the same P2P event from reaching more than one run listener.

## Offline and live verification

The offline suite covers:

- five sessions accepted and the sixth rejected;
- unlimited multi-turn transcript persistence and crash reload;
- sender filtering and event deduplication;
- close request, rejection, and explicit confirmation;
- stable structured outcomes;
- allowlisted/bounded/redacted evidence and verifier diffs;
- safe `lark-cli` argv construction without a shell;
- long-connection ready handshake and event normalization;
- durable outbox recovery;
- task checkpoint ordering and resume behavior;
- evaluator freeze, explicit approval, held guidance, and disabled-mode compatibility.
- model-backed Human Proxy isolation from solution/evaluator execution, transcript-driven multi-turn dialogue, close rejection/continuation, crash recovery, five-session budget, and approve-only gated install.

Before declaring a deployment ready, perform one real Feishu session on the target machine: open it, exchange several ordinary turns, ask for a run-evidence detail, request closure, reject closure once, continue talking, request closure again, explicitly confirm, then inspect `transcript.jsonl`, `outcome.json`, `index.json`, and `events.jsonl`.
