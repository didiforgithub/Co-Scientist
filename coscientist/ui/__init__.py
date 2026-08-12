"""Co-Scientist web UI (Task 2) — human interaction + visualization.

STDLIB-ONLY, ADDITIVE package. Nothing here imports flask/fastapi or any
third-party dependency; the server is built on ``http.server``. It never edits
existing source — it only READS the on-disk ``runs/<id>/`` trees (see
``coevo.store``) and exposes a human-review control plane (``WebHumanPort``,
the web equivalent of ``demo.human_port.CliHuman``).

Launch::

    PYTHONPATH=. python3 -m coscientist.ui.server --runs-dir runs --port 8765
"""

from .review_registry import PendingReview, ReviewRegistry

__all__ = ["ReviewRegistry", "PendingReview", "WebHumanPort"]


def __getattr__(name):
    # Lazy so importing the (stdlib-only) server never eagerly pulls WebHumanPort's
    # transitive numpy dependency (via demo.human_port dataclasses). The viz server
    # thus starts even on a box without numpy; WebHumanPort loads on first use.
    if name == "WebHumanPort":
        from .web_human_port import WebHumanPort
        return WebHumanPort
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
