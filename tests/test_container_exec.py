from __future__ import annotations

import subprocess
from pathlib import Path

from coscientist.coevo.container import DockerContainer, GatewayConfig


def _live_container(tmp_path: Path) -> DockerContainer:
    container = DockerContainer(
        workdir=tmp_path,
        gateway=GatewayConfig(codex_home=tmp_path / "codex-home"),
        agent_elf=tmp_path / "codex",
    )
    container._started = True
    container._cid = "solver-container"
    return container


def test_exec_agent_closes_stdin_instead_of_attaching_it(tmp_path, monkeypatch):
    calls: list[tuple[list[str], dict]] = []

    def fake_run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    session = _live_container(tmp_path).exec_agent("solve it", timeout_s=30)

    assert session.ok
    agent_argv, agent_kwargs = next(
        (argv, kwargs) for argv, kwargs in calls if "codex" in argv
    )
    assert "-i" not in agent_argv
    assert agent_kwargs["stdin"] is subprocess.DEVNULL


def test_exec_agent_timeout_kills_the_in_container_codex(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        argv = list(argv)
        calls.append(argv)
        if "codex" in argv:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    session = _live_container(tmp_path).exec_agent("solve it", timeout_s=1)

    assert not session.ok
    assert session.note == "agent hit the wall-clock budget"
    cleanup = [argv for argv in calls if "kill -TERM" in " ".join(argv)]
    assert len(cleanup) == 1
    cleanup_text = " ".join(cleanup[0])
    assert "solver-container" in cleanup_text
    assert "kill -KILL" in cleanup_text
