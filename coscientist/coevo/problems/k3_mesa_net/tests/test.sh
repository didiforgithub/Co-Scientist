#!/usr/bin/env bash
set -euo pipefail

mkdir -p /logs/verifier

write_zero() {
    local reason="$1"
    python3 - "$reason" <<'PY'
import json
import sys
from pathlib import Path
result = {"reward": 0.0, "correct": False, "reason": sys.argv[1]}
Path("/logs/verifier/reward.json").write_text(json.dumps(result, indent=2) + "\n")
Path("/logs/verifier/reward.txt").write_text("0.0")
PY
}

if ! sha256sum -c /orig/protected.sha256 --quiet; then
    write_zero "protected_file_modified"
    exit 0
fi

if ! python3 /orig/scorer.py \
    --reference /orig/reference_model.py \
    --candidate /app/candidate.py \
    --headroom 63.5 \
    --hidden \
    --correctness-seeds 2 \
    --warmup 3 \
    --iterations 10 \
    --json-out /logs/verifier/reward.json \
    --reward-out /logs/verifier/reward.txt; then
    write_zero "verifier_process_failed"
fi

exit 0
