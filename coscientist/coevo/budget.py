"""Wall-clock budget (§5) — the single outer bound on a co-evolution run.

The reference demo bounds work by ``outer_rounds × inner_steps``. The target
bounds it by a **wall clock**: at most two days (the user's ceiling). This module
is the one place that decides "is there time left?", behind an injectable clock so
tests run instantly and deterministically without sleeping.

    deadline = Deadline.days(2)          # real wall clock
    deadline = Deadline(budget_s=100, clock=fake.tick)   # test: drive time by hand

Both the Solver's inner loop and the driver's event loop consult the SAME
deadline, so the whole system stops together when the budget is spent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

TWO_DAYS_S = 2 * 24 * 60 * 60  # the ceiling the user set: "最多两天"


@dataclass
class Deadline:
    """A wall-clock budget with an injectable monotonic clock.

    ``clock`` returns seconds (any monotonic source). ``budget_s`` is how long the
    run may last from construction. Everything is derived from ``clock`` so a test
    can advance time explicitly and never actually sleep.
    """

    budget_s: float = TWO_DAYS_S
    clock: Callable[[], float] = time.monotonic
    _t0: float = field(init=False)

    def __post_init__(self) -> None:
        self._t0 = self.clock()

    @classmethod
    def days(cls, n: float, *, clock: Callable[[], float] = time.monotonic) -> "Deadline":
        return cls(budget_s=n * 24 * 60 * 60, clock=clock)

    def elapsed(self) -> float:
        return self.clock() - self._t0

    def remaining(self) -> float:
        return max(0.0, self.budget_s - self.elapsed())

    def expired(self) -> bool:
        return self.elapsed() >= self.budget_s

    def fraction(self) -> float:
        """Elapsed / budget, clamped to [0, 1] — the EEH wall-clock x-axis (§5)."""
        if self.budget_s <= 0:
            return 1.0
        return min(1.0, self.elapsed() / self.budget_s)
