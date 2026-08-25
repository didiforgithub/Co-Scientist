"""Read-only evidence packets and a session-scoped Codex expert liaison.

The liaison never receives a mount of the live solver/evaluator workspace.  It
gets an allowlisted, size-bounded copy of run evidence and may write only its own
``response.json``.  This keeps a conversational agent useful without allowing a
human session to mutate the evaluator behind the orchestrator's safety gate.
"""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .container import AgentSession, GatewayConfig, one_shot_agent
from .human_sessions import HumanSession, SessionMessage, SessionOutcome

_SECRET_KEY = re.compile(
    r"(^|_)(api_?key|secret|password|passwd|token|authorization|credential)s?($|_)",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"(?i)(bearer\s+[a-z0-9._~+/-]{12,}|sk-[a-z0-9_-]{8,}|"
    r"(?:api[_-]?key|secret|password|token)\s*[=:]\s*[^\s,;]+)"
)


class EvidenceAgentError(RuntimeError):
    pass


@dataclass(frozen=True)
class EvidenceReply:
    text: str
    proposed_outcome: SessionOutcome | None = None


class EvidenceAgent(Protocol):
    def reply(
        self,
        *,
        session: HumanSession,
        transcript: Sequence[SessionMessage],
        human_message: str,
        agent_instruction: str,
    ) -> EvidenceReply: ...


class RunEvidenceBuilder:
    """Build a bounded packet from a deliberately small run-file allowlist."""

    TASK_FILES = ("instruction.md", "task.toml", "resource.toml")

    def __init__(
        self,
        run_dir: Path | str,
        raw_input_dir: Path | str,
        *,
        max_total_chars: int = 100_000,
        max_file_chars: int = 20_000,
        tail_records: int = 20,
    ):
        if max_total_chars < 512:
            raise ValueError("max_total_chars must be at least 512")
        self.run_dir = Path(run_dir)
        self.raw_input_dir = Path(raw_input_dir)
        self.max_total_chars = max_total_chars
        self.max_file_chars = max_file_chars
        self.tail_records = tail_records

    def build(self) -> dict[str, Any]:
        task = {
            name: self._read_text(self.raw_input_dir / name)
            for name in self.TASK_FILES
            if (self.raw_input_dir / name).is_file()
        }
        snapshot: dict[str, Any] = {
            "task": task,
            "manifest": self._read_json(self.run_dir / "manifest.json"),
            "events": self._read_jsonl_tail(self.run_dir / "events.jsonl"),
            "trajectory": self._read_jsonl_tail(
                self.run_dir / "solver" / "trajectory.jsonl"
            ),
            "eval_queries": self._read_jsonl_tail(
                self.run_dir / "eval" / "queries.jsonl"
            ),
            "supervisor_reviews": self._read_jsonl_tail(
                self.run_dir / "supervisor" / "reviews.jsonl"
            ),
            "verifier_versions": self._read_jsonl_tail(
                self.run_dir / "supervisor" / "versions.jsonl"
            ),
            "candidates": self._recent_json_files(
                self.run_dir / "solver" / "candidates", "cand_*.json"
            ),
            "probes": self._recent_json_files(
                self.run_dir / "supervisor" / "probes", "probe_*.json"
            ),
            "verifier": self._verifier_evidence(),
            "limits": {
                "tail_records": self.tail_records,
                "max_file_chars": self.max_file_chars,
                "max_total_chars": self.max_total_chars,
            },
        }
        return self._sanitize(snapshot)

    def render(self, snapshot: dict[str, Any] | None = None) -> str:
        """Return valid JSON no larger than ``max_total_chars``."""
        payload = self._sanitize(snapshot if snapshot is not None else self.build())
        rendered = json.dumps(payload, indent=2, ensure_ascii=False)
        if len(rendered) <= self.max_total_chars:
            return rendered
        # Keep the packet valid and explicit about truncation.  The excerpt remains
        # useful to the agent while the limit is a hard guarantee, not a best effort.
        low, high = 0, len(rendered)
        result = "{}"
        while low <= high:
            middle = (low + high) // 2
            candidate = json.dumps(
                {
                    "truncated": True,
                    "reason": "evidence packet exceeded max_total_chars",
                    "evidence_json_excerpt": rendered[:middle],
                },
                ensure_ascii=False,
            )
            if len(candidate) <= self.max_total_chars:
                result = candidate
                low = middle + 1
            else:
                high = middle - 1
        return result

    def sanitize(self, value: Any) -> Any:
        """Redact secret-shaped keys/values in checkpoint context as well as files."""
        return self._sanitize(value)

    def _verifier_evidence(self) -> dict[str, Any]:
        directory = self.run_dir / "supervisor" / "verifier_versions"
        if not directory.is_dir():
            return {}

        def version_number(path: Path) -> int:
            match = re.fullmatch(r"v(\d+)\.py", path.name)
            return int(match.group(1)) if match else -1

        versions = sorted(
            (path for path in directory.glob("v*.py") if version_number(path) >= 0),
            key=version_number,
        )
        if not versions:
            return {}
        current = versions[-1]
        current_source = self._read_text(current)
        result: dict[str, Any] = {
            "current_version": version_number(current),
            "current_source": current_source,
        }
        if len(versions) > 1:
            previous = versions[-2]
            previous_source = self._read_text(previous)
            result["previous_version"] = version_number(previous)
            result["diff_from_previous"] = "".join(
                difflib.unified_diff(
                    previous_source.splitlines(keepends=True),
                    current_source.splitlines(keepends=True),
                    fromfile=previous.name,
                    tofile=current.name,
                )
            )[: self.max_file_chars]
        return result

    def _read_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")[: self.max_file_chars]
        except OSError:
            return ""

    def _read_json(self, path: Path) -> dict[str, Any]:
        try:
            value = json.loads(self._read_text(path))
        except (json.JSONDecodeError, OSError):
            return {}
        return value if isinstance(value, dict) else {"value": value}

    def _read_jsonl_tail(self, path: Path) -> list[Any]:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()[-self.tail_records :]
        except OSError:
            return []
        records: list[Any] = []
        for line in lines:
            try:
                records.append(json.loads(line[: self.max_file_chars]))
            except json.JSONDecodeError:
                records.append({"unparsed": line[: self.max_file_chars]})
        return records

    def _recent_json_files(self, directory: Path, pattern: str) -> list[Any]:
        if not directory.is_dir():
            return []
        paths = sorted(directory.glob(pattern))[-self.tail_records :]
        return [self._read_json(path) for path in paths]

    def _sanitize(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): (
                    "[REDACTED]" if _SECRET_KEY.search(str(key)) else self._sanitize(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._sanitize(item) for item in value]
        if isinstance(value, tuple):
            return [self._sanitize(item) for item in value]
        if isinstance(value, str):
            return _SECRET_VALUE.sub("[REDACTED]", value)
        return value


Runner = Callable[..., AgentSession]


class CodexEvidenceAgent:
    """A fresh, session-scoped Codex turn over an isolated evidence packet."""

    def __init__(
        self,
        run_dir: Path | str,
        raw_input_dir: Path | str,
        *,
        gateway: GatewayConfig,
        agent_elf: Path,
        image: str = "python:3.11-slim",
        timeout_s: float = 180.0,
        runner: Runner | None = None,
    ):
        self.run_dir = Path(run_dir)
        self.raw_input_dir = Path(raw_input_dir)
        self.gateway = gateway
        self.agent_elf = Path(agent_elf)
        self.image = image
        self.timeout_s = timeout_s
        self.runner = runner

    def reply(
        self,
        *,
        session: HumanSession,
        transcript: Sequence[SessionMessage],
        human_message: str,
        agent_instruction: str,
    ) -> EvidenceReply:
        workspace = (
            self.run_dir
            / "human"
            / session.session_id
            / "agent_workspace"
        )
        workspace.mkdir(parents=True, exist_ok=True)
        response_path = workspace / "response.json"
        response_path.unlink(missing_ok=True)

        builder = RunEvidenceBuilder(self.run_dir, self.raw_input_dir)
        snapshot = builder.build()
        (workspace / "context.json").write_text(
            builder.render(snapshot), encoding="utf-8"
        )
        transcript_payload = [asdict(item) for item in transcript]
        (workspace / "transcript.json").write_text(
            json.dumps(transcript_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        safe_session = HumanSession(
            **{
                **asdict(session),
                "state": session.state,
                "context": builder.sanitize(session.context),
            }
        )
        prompt = self._prompt(
            session=safe_session,
            transcript=transcript,
            human_message=human_message,
            agent_instruction=agent_instruction,
            evidence_brief=self._evidence_brief(snapshot),
        )
        result = self._run(workspace, prompt)
        if not result.ok:
            raise EvidenceAgentError(
                f"evidence agent failed: {result.note or result.stderr[-300:]}"
            )
        if not response_path.is_file():
            raise EvidenceAgentError("evidence agent did not write response.json")
        try:
            payload = json.loads(response_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise EvidenceAgentError("evidence agent wrote invalid response.json") from exc
        reply = str(payload.get("reply", "")).strip()
        if not reply:
            raise EvidenceAgentError("evidence agent response is missing reply text")
        raw_outcome = payload.get("proposed_outcome")
        outcome = (
            SessionOutcome.from_dict(raw_outcome)
            if isinstance(raw_outcome, dict)
            else None
        )
        return EvidenceReply(text=reply, proposed_outcome=outcome)

    def _run(self, workspace: Path, prompt: str) -> AgentSession:
        if self.runner is not None:
            return self.runner(
                workspace,
                prompt,
                gateway=self.gateway,
                agent_elf=self.agent_elf,
                timeout_s=self.timeout_s,
                image=self.image,
            )
        return one_shot_agent(
            workspace,
            self.gateway,
            self.agent_elf,
            prompt,
            timeout_s=self.timeout_s,
            image=self.image,
            disallowed_tools=["network"],
        )

    @staticmethod
    def _prompt(
        *,
        session: HumanSession,
        transcript: Sequence[SessionMessage],
        human_message: str,
        agent_instruction: str,
        evidence_brief: str,
    ) -> str:
        transcript_text = "\n".join(
            f"[{item.role}] {item.text}" for item in transcript
        ) or "(本轮此前没有消息)"
        outcome_shape = json.dumps(SessionOutcome().to_dict(), ensure_ascii=False)
        return f"""你是 Co-Scientist 在一次稀缺人类专家会话中的 Supervisor。

目标：像正常同事聊天一样，直接、简洁地回答专家；必要时主动检查证据，暴露系统尚未弄清楚的任务约束。不要要求专家使用命令。

安全边界：
- `context.json` 是从真实 run 复制出的只读、白名单证据；`transcript.json` 是完整对话。
- 回答前必须先读取并核对 `/work/context.json` 和 `/work/transcript.json`。下方证据摘要只是索引；需要细节时必须回到文件检查。
- `context.task` 非空时，不得声称“没有具体任务”或要求人类重复提供文件中已有的目标、shape、dtype、评分或约束；应先用现有证据直接回答，并明确区分已知与未知。
- 不得声称看过 context 中不存在的证据，不得访问网络，不得修改 solver、verifier 或 run 状态。
- 你唯一应写的文件是 `/work/response.json`。
- 如果证据不足，明确说不知道，并具体说明还需要什么 probe/数据。

经过脱敏和长度限制的证据摘要：
{evidence_brief}

会话目的：{session.purpose}
本会话检查点上下文：
{json.dumps(session.context, indent=2, ensure_ascii=False)}

编排器给你的本轮指令：{agent_instruction}

此前对话：
{transcript_text}

专家最新消息：
{human_message}

写入 `/work/response.json`，格式严格为：
{{
  "reply": "发给专家的自然语言回复",
  "proposed_outcome": null
}}

只有编排器要求你总结并请求关闭时，才把 `proposed_outcome` 写成对象；字段形状为：
{outcome_shape}
`decision` 只能是 approve/reject/guide/none。这个 outcome 只是等待专家确认的草案，不能据此自行改变 evaluator。
"""

    @staticmethod
    def _evidence_brief(snapshot: dict[str, Any]) -> str:
        task = snapshot.get("task", {})
        task_excerpt = (
            {
                str(name): str(content)[:6_000]
                for name, content in task.items()
            }
            if isinstance(task, dict)
            else {}
        )
        inventory: dict[str, Any] = {}
        for key in (
            "events",
            "trajectory",
            "eval_queries",
            "supervisor_reviews",
            "verifier_versions",
            "candidates",
            "probes",
            "verifier",
        ):
            value = snapshot.get(key)
            if isinstance(value, (list, dict)):
                inventory[key] = len(value)
            else:
                inventory[key] = bool(value)
        return json.dumps(
            {"task": task_excerpt, "evidence_inventory": inventory},
            ensure_ascii=False,
            indent=2,
        )[:20_000]
