"""Small `lark-cli` transport for Feishu direct-message Human Sessions."""

from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any


class LarkCliError(RuntimeError):
    pass


@dataclass(frozen=True)
class FeishuMessageEvent:
    event_id: str
    message_id: str
    sender_id: str
    chat_id: str
    chat_type: str
    message_type: str
    text: str
    timestamp: str


class LarkCliTransport:
    EVENT_KEY = "im.message.receive_v1"

    def __init__(
        self,
        *,
        executable: str = "lark-cli",
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        process_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
        command_timeout_s: float = 30.0,
    ):
        self.executable = executable
        self.runner = runner
        self.process_factory = process_factory
        self.command_timeout_s = command_timeout_s

    def send_dm(self, *, user_id: str, text: str, idempotency_key: str) -> str:
        return self._message_command(
            [
                self.executable,
                "im",
                "+messages-send",
                "--as",
                "bot",
                "--user-id",
                user_id,
                "--text",
                text,
                "--idempotency-key",
                idempotency_key,
                "--format",
                "json",
            ]
        )

    def reply(self, *, message_id: str, text: str, idempotency_key: str) -> str:
        return self._message_command(
            [
                self.executable,
                "im",
                "+messages-reply",
                "--as",
                "bot",
                "--message-id",
                message_id,
                "--text",
                text,
                "--idempotency-key",
                idempotency_key,
                "--format",
                "json",
            ]
        )

    def _message_command(self, argv: list[str]) -> str:
        try:
            result = self.runner(
                argv,
                shell=False,
                capture_output=True,
                text=True,
                timeout=self.command_timeout_s,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise LarkCliError(f"lark-cli command failed: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown lark-cli error").strip()
            raise LarkCliError(detail[-600:])
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise LarkCliError("lark-cli returned non-JSON message output") from exc
        message_id = self._find_message_id(payload)
        if not message_id:
            raise LarkCliError("lark-cli response did not contain a message_id")
        return message_id

    @classmethod
    def _find_message_id(cls, value: Any) -> str:
        if isinstance(value, dict):
            for key in ("message_id", "messageId"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate:
                    return candidate
            for item in value.values():
                found = cls._find_message_id(item)
                if found:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = cls._find_message_id(item)
                if found:
                    return found
        return ""

    @classmethod
    def normalize_event(cls, payload: dict[str, Any]) -> FeishuMessageEvent:
        """Normalize both lark-cli's flat schema and Feishu's nested event shape."""
        body = payload.get("event", payload)
        message = body.get("message", body)
        sender = body.get("sender", {})
        sender_id_value = body.get("sender_id")
        if not sender_id_value and isinstance(sender, dict):
            sender_id_value = sender.get("sender_id", sender.get("id", ""))
            if isinstance(sender_id_value, dict):
                sender_id_value = (
                    sender_id_value.get("open_id")
                    or sender_id_value.get("user_id")
                    or ""
                )
        content: Any = message.get("content", body.get("content", ""))
        if isinstance(content, str):
            try:
                decoded = json.loads(content)
                content = decoded
            except json.JSONDecodeError:
                pass
        text = content.get("text", "") if isinstance(content, dict) else str(content)
        header = payload.get("header", {}) if isinstance(payload.get("header"), dict) else {}
        event_id = str(
            body.get("event_id")
            or payload.get("event_id")
            or header.get("event_id")
            or ""
        )
        message_id = str(
            message.get("message_id")
            or message.get("id")
            or body.get("message_id")
            or body.get("id")
            or ""
        )
        if not event_id or not message_id or not sender_id_value:
            raise LarkCliError("message event is missing event_id, message_id, or sender_id")
        return FeishuMessageEvent(
            event_id=event_id,
            message_id=message_id,
            sender_id=str(sender_id_value),
            chat_id=str(message.get("chat_id") or body.get("chat_id") or ""),
            chat_type=str(message.get("chat_type") or body.get("chat_type") or ""),
            message_type=str(
                message.get("message_type") or body.get("message_type") or ""
            ),
            text=str(text),
            timestamp=str(
                body.get("timestamp")
                or message.get("create_time")
                or payload.get("timestamp")
                or ""
            ),
        )

    def consume_events(self, *, ready_timeout_s: float = 15.0) -> Iterator[FeishuMessageEvent]:
        """Yield text events after the long-connection consumer reports readiness."""
        argv = [
            self.executable,
            "event",
            "consume",
            self.EVENT_KEY,
            "--as",
            "bot",
        ]
        try:
            process = self.process_factory(
                argv,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise LarkCliError(f"could not start lark-cli event consumer: {exc}") from exc
        if process.stdout is None or process.stderr is None:
            raise LarkCliError("lark-cli event consumer has no output pipes")

        ready = threading.Event()
        stderr_tail: list[str] = []

        def drain_stderr() -> None:
            for line in process.stderr:
                stripped = line.strip()
                if stripped:
                    stderr_tail.append(stripped)
                    del stderr_tail[:-20]
                if "[event] ready" in stripped and self.EVENT_KEY in stripped:
                    ready.set()

        stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
        stderr_thread.start()
        if not ready.wait(timeout=ready_timeout_s):
            self._terminate(process)
            detail = stderr_tail[-1] if stderr_tail else "ready marker not received"
            raise LarkCliError(f"Feishu event consumer did not become ready: {detail}")
        try:
            for line in process.stdout:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    yield self.normalize_event(payload)
                except (json.JSONDecodeError, LarkCliError):
                    # A malformed/non-message event must not bring down the durable
                    # listener; it cannot be routed safely, so skip it.
                    continue
        finally:
            self._terminate(process)

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            # Do not SIGKILL the shared lark event daemon.  Closing this consumer's
            # pipes is sufficient; the daemon is intentionally allowed to survive.
            for stream in (process.stdout, process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass
