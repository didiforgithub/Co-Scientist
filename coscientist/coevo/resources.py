"""First-class resource configuration for the co-evolution run.

A task may ship a resource config alongside its instruction/files. We model it as
a single abstraction — a ``ResourceSpec`` with TWO isolated slices:

    ResourceSpec
      .solver    -> the long-lived Solver container's resources
      .verifier  -> the Eval/verify container's resources (GPU-backed checker seam)

The two slices are deliberately independent (Harbor-style ``[environment]`` vs
``[verifier.environment]`` split): the Solver and the grader must not share a
GPU/CPU/memory budget or a network posture. Isolation is EXPLICIT — the verifier
slice is never inferred from the solver slice; a config must declare it to get it.

Sources are layered, later wins::

    builtin defaults  <  resource.toml (or a K3 task.toml)  <  CLI/web overrides

An empty spec (no file, no overrides) means "current behavior": a plain Solver
container (no --gpus/--cpus/...) and host-subprocess verify. Every field is
Optional so "unset" is distinguishable from "set to a default" and merging is a
simple non-None-wins fold.

The loader is deliberately tolerant: a missing file yields defaults; a malformed
file yields defaults plus a recorded ``note`` (it never raises into a run).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

try:  # py3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - older interpreters
    import tomli as _toml  # type: ignore


# Recognised resource keys on a single container slice. Kept small and explicit
# so an unknown key in the file is ignored rather than silently mis-mapped.
_SLICE_KEYS = ("image", "cpus", "memory_mb", "gpus", "gpu_types",
               "timeout_sec", "allow_internet")


def _as_gpus(value: Any) -> Optional[str]:
    """Normalise a gpus value to the ``docker run --gpus`` spec string.

    Accepts an int count (``1`` -> ``"1"``), ``"all"``, or an already-formed spec.
    ``0``/``None``/``""`` -> None (no GPU request)."""
    if value is None:
        return None
    if isinstance(value, bool):  # guard: True/False is not a gpu count
        return None
    if isinstance(value, (int, float)):
        n = int(value)
        return str(n) if n > 0 else None
    s = str(value).strip()
    return s or None


@dataclass
class ContainerResources:
    """Resources for one container (a Solver slice or a Verifier slice).

    Every field Optional so "unset" (inherit/ignore) differs from a concrete value.
    ``allow_internet`` defaults to False — the safe posture — but only takes effect
    (as ``--network none``) when the slice is otherwise active; see
    ``DockerContainer`` / the verify backend.
    """

    image: Optional[str] = None
    cpus: Optional[float] = None
    memory_mb: Optional[int] = None
    gpus: Optional[str] = None
    gpu_types: list[str] = field(default_factory=list)
    timeout_sec: Optional[float] = None
    allow_internet: bool = False

    def is_empty(self) -> bool:
        """True when nothing was configured — the default/no-op slice."""
        return (self.image is None and self.cpus is None and self.memory_mb is None
                and self.gpus is None and not self.gpu_types
                and self.timeout_sec is None and self.allow_internet is False)

    def merge(self, other: "ContainerResources") -> "ContainerResources":
        """Return self overlaid with ``other`` (other's non-None fields win).

        ``gpu_types`` merges by replacement when ``other`` provides a non-empty
        list. ``allow_internet`` is a bool so the caller controls precedence by
        only setting it on the overriding layer (see ``merge_overrides``)."""
        return ContainerResources(
            image=other.image if other.image is not None else self.image,
            cpus=other.cpus if other.cpus is not None else self.cpus,
            memory_mb=other.memory_mb if other.memory_mb is not None else self.memory_mb,
            gpus=other.gpus if other.gpus is not None else self.gpus,
            gpu_types=list(other.gpu_types) if other.gpu_types else list(self.gpu_types),
            timeout_sec=other.timeout_sec if other.timeout_sec is not None else self.timeout_sec,
            allow_internet=other.allow_internet or self.allow_internet,
        )

    def to_manifest(self) -> dict:
        """A JSON-friendly view for the manifest / UI (drops unset fields)."""
        out: dict[str, Any] = {}
        if self.image is not None:
            out["image"] = self.image
        if self.cpus is not None:
            out["cpus"] = self.cpus
        if self.memory_mb is not None:
            out["memory_mb"] = self.memory_mb
        if self.gpus is not None:
            out["gpus"] = self.gpus
        if self.gpu_types:
            out["gpu_types"] = list(self.gpu_types)
        if self.timeout_sec is not None:
            out["timeout_sec"] = self.timeout_sec
        out["allow_internet"] = self.allow_internet
        return out

    @classmethod
    def from_table(cls, table: dict) -> "ContainerResources":
        """Build a slice from a parsed TOML table, tolerating unknown keys."""
        if not isinstance(table, dict):
            return cls()
        return cls(
            image=(str(table["image"]) if table.get("image") is not None else None),
            cpus=(float(table["cpus"]) if table.get("cpus") is not None else None),
            memory_mb=(int(table["memory_mb"]) if table.get("memory_mb") is not None else None),
            gpus=_as_gpus(table.get("gpus")),
            gpu_types=[str(x) for x in (table.get("gpu_types") or [])],
            timeout_sec=(float(table["timeout_sec"]) if table.get("timeout_sec") is not None else None),
            allow_internet=bool(table.get("allow_internet", False)),
        )


@dataclass
class ResourceSpec:
    """The two isolated slices that drive the two containers.

    ``note`` records loader diagnostics (source file, parse fallback) for the
    manifest — never raised.
    """

    solver: ContainerResources = field(default_factory=ContainerResources)
    verifier: ContainerResources = field(default_factory=ContainerResources)
    note: str = ""

    @classmethod
    def defaults(cls) -> "ResourceSpec":
        """The no-op spec: both slices empty ⇒ today's behavior everywhere."""
        return cls()

    def merge_overrides(self, *, solver: Optional[ContainerResources] = None,
                        verifier: Optional[ContainerResources] = None) -> "ResourceSpec":
        """Overlay CLI/web override slices (their non-None fields win)."""
        return replace(
            self,
            solver=self.solver.merge(solver) if solver is not None else self.solver,
            verifier=self.verifier.merge(verifier) if verifier is not None else self.verifier,
        )

    def to_manifest(self) -> dict:
        out = {"solver": self.solver.to_manifest(),
               "verifier": self.verifier.to_manifest()}
        if self.note:
            out["note"] = self.note
        return out


# Candidate filenames, in priority order, discovered inside a raw-input dir.
_RESOURCE_FILENAMES = ("resource.toml", "resources.toml", "task.toml")


def find_resource_file(raw_input_dir: Path) -> Optional[Path]:
    """First existing resource file in ``raw_input_dir`` by priority, else None."""
    d = Path(raw_input_dir)
    for name in _RESOURCE_FILENAMES:
        p = d / name
        if p.is_file():
            return p
    return None


def _spec_from_tables(data: dict) -> ResourceSpec:
    """Map a parsed TOML document to a ResourceSpec (Harbor-aligned).

    Precedence within one file, per slice:
      solver   <- [solver], else K3 [environment] (+ [agent].timeout_sec)
      verifier <- [verifier.environment] merged over [verifier]; K3 [verifier]
                  only contributes timeout_sec unless it declares resources.

    A file that only carries a single [environment] block seeds the SOLVER slice;
    the verifier stays default — isolation is explicit, never inferred.
    """
    solver = ContainerResources()
    verifier = ContainerResources()

    env = data.get("environment")
    if isinstance(env, dict):
        solver = solver.merge(ContainerResources.from_table(env))
    sol_tbl = data.get("solver")
    if isinstance(sol_tbl, dict):
        solver = solver.merge(ContainerResources.from_table(sol_tbl))
    # K3 [agent].timeout_sec -> solver.timeout_sec (only if not already set)
    agent = data.get("agent")
    if isinstance(agent, dict) and agent.get("timeout_sec") is not None and solver.timeout_sec is None:
        solver = solver.merge(ContainerResources(timeout_sec=float(agent["timeout_sec"])))

    ver = data.get("verifier")
    if isinstance(ver, dict):
        # [verifier] scalar resources (K3 mainly gives timeout_sec here)
        verifier = verifier.merge(ContainerResources.from_table(ver))
        ver_env = ver.get("environment")
        if isinstance(ver_env, dict):
            verifier = verifier.merge(ContainerResources.from_table(ver_env))

    return ResourceSpec(solver=solver, verifier=verifier)


def load(raw_input_dir: Path, *, explicit_path: Optional[Path] = None) -> ResourceSpec:
    """Load a ResourceSpec for a run.

    ``explicit_path`` (a --resource-config CLI value) wins over auto-discovery in
    ``raw_input_dir``. Missing/unparseable ⇒ ``defaults()`` (with a ``note``);
    never raises.
    """
    path: Optional[Path] = None
    if explicit_path is not None:
        p = Path(explicit_path)
        if p.is_file():
            path = p
        else:
            spec = ResourceSpec.defaults()
            spec.note = f"resource-config not found: {p}"
            return spec
    else:
        path = find_resource_file(raw_input_dir)

    if path is None:
        return ResourceSpec.defaults()

    try:
        with open(path, "rb") as fh:
            data = _toml.load(fh)
    except Exception as e:  # noqa: BLE001 — a bad file must not break a run
        spec = ResourceSpec.defaults()
        spec.note = f"resource file {path.name} unparseable ({e}); using defaults"
        return spec

    spec = _spec_from_tables(data if isinstance(data, dict) else {})
    spec.note = f"loaded from {path.name}"
    return spec
