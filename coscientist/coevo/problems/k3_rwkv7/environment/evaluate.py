#!/usr/bin/env python3
import subprocess
import sys

command = [
    sys.executable,
    "/app/scorer.py",
    "--reference", "/app/reference_model.py",
    "--candidate", "/app/candidate.py",
    "--headroom", "80",
    "--seed", "0",
    "--correctness-seeds", "1",
]
if "--full" not in sys.argv[1:]:
    command.append("--quick")
raise SystemExit(subprocess.call(command))
