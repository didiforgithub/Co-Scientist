#!/usr/bin/env python3
"""Forward-correctness and latency scorer for the 100-task kernel suite."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import secrets
import sys
import time
import traceback
from pathlib import Path

import torch


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pick_output(value, index: int):
    return value[index] if isinstance(value, (tuple, list)) else value


def move_inputs(values, device: str):
    return [value.to(device) if torch.is_tensor(value) else value for value in values]


def check_correctness(ref_mod, candidate_cls, shape, device, seed, rtol, atol):
    init_args = ref_mod.get_init_inputs()
    reference = ref_mod.Model(*init_args).to(device)
    candidate = candidate_cls(*init_args).to(device)
    ref_inputs = move_inputs(ref_mod.make_inputs(shape, device, seed), device)
    cand_inputs = move_inputs(ref_mod.make_inputs(shape, device, seed), device)
    output_index = getattr(ref_mod, "OUTPUT_INDEX", 0)
    with torch.no_grad():
        ref_output = pick_output(reference(*ref_inputs), output_index)
        cand_output = pick_output(candidate(*cand_inputs), output_index)
    ref_float = ref_output.float()
    cand_float = cand_output.float()
    correct = bool(
        ref_float.shape == cand_float.shape
        and torch.isfinite(cand_float).all()
        and torch.allclose(ref_float, cand_float, rtol=rtol, atol=atol)
    )
    max_diff = float((ref_float - cand_float).abs().max().item()) if ref_float.shape == cand_float.shape else None
    del reference, candidate, ref_inputs, cand_inputs, ref_output, cand_output
    gc.collect()
    torch.cuda.empty_cache()
    return correct, max_diff


def measure(ref_mod, model_cls, shape, device, seed, warmup, iterations):
    model = model_cls(*ref_mod.get_init_inputs()).to(device)
    inputs = move_inputs(ref_mod.make_inputs(shape, device, seed), device)
    output_index = getattr(ref_mod, "OUTPUT_INDEX", 0)

    def step():
        with torch.no_grad():
            pick_output(model(*inputs), output_index)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    step()
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6

    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        step()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0 / iterations

    del model, inputs
    gc.collect()
    torch.cuda.empty_cache()
    return peak_mb, elapsed_ms


def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def score(args):
    if not torch.cuda.is_available():
        return {"reward": 0.0, "correct": False, "reason": "CUDA is not available"}

    device = f"cuda:{args.device}"
    torch.cuda.set_device(args.device)
    sys.path.insert(0, str(Path(args.candidate).resolve().parent))
    ref_mod = load_module(args.reference, "kernel_reference_oracle")
    cand_mod = load_module(args.candidate, "kernel_candidate")
    if not hasattr(cand_mod, "ModelNew"):
        return {"reward": 0.0, "correct": False, "reason": "candidate.py must define ModelNew"}

    workload = list(ref_mod.WORKLOAD)
    if args.quick:
        workload = workload[:1]
    rtol = float(getattr(ref_mod, "TASK_RTOL", args.rtol))
    atol = float(getattr(ref_mod, "TASK_ATOL", args.atol))
    warmup = 1 if args.quick else args.warmup
    iterations = 1 if args.quick else args.iterations
    base_seed = secrets.randbits(30) if args.hidden else args.seed
    correctness_seeds = [base_seed + offset for offset in range(args.correctness_seeds)]
    measurement_seed = base_seed + args.correctness_seeds + 17

    rows = []
    all_correct = True
    for shape in workload:
        row = {"shape": list(shape)}
        try:
            checks = []
            for seed in correctness_seeds:
                ok, max_diff = check_correctness(
                    ref_mod, cand_mod.ModelNew, shape, device, seed, rtol, atol
                )
                checks.append(
                    {
                        "seed": "hidden" if args.hidden else seed,
                        "correct": ok,
                        "max_abs_diff": max_diff,
                    }
                )
                all_correct = all_correct and ok
            candidate_memory, candidate_ms = measure(
                ref_mod,
                cand_mod.ModelNew,
                shape,
                device,
                measurement_seed,
                warmup,
                iterations,
            )
            reference_memory, reference_ms = measure(
                ref_mod,
                ref_mod.Model,
                shape,
                device,
                measurement_seed,
                warmup,
                iterations,
            )
            speedup = reference_ms / candidate_ms
            row.update(
                correct=all(check["correct"] for check in checks),
                checks=checks,
                candidate_ms=round(candidate_ms, 4),
                reference_ms=round(reference_ms, 4),
                candidate_peak_mb=round(candidate_memory, 2),
                reference_peak_mb=round(reference_memory, 2),
                speedup_vs_naive=round(speedup, 5),
            )
        except Exception as exc:
            all_correct = False
            row.update(
                correct=False,
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc()[-1200:],
            )
        rows.append(row)

    speedups = [row["speedup_vs_naive"] for row in rows if row.get("correct") and row.get("speedup_vs_naive", 0) > 0]
    improvement = geometric_mean(speedups) if len(speedups) == len(rows) and speedups else 0.0
    if not all_correct or improvement <= 1.0 or args.headroom <= 1.0:
        reward = 0.0
    else:
        reward = min(1.0, 0.5 * math.log(improvement) / math.log(args.headroom))
    return {
        "reward": round(max(0.0, reward), 4),
        "correct": bool(all_correct),
        "score": round(improvement if all_correct else 0.0, 5),
        "metric": "naive_relative_speedup",
        "naive_anchor": 1.0,
        "strong_baseline_anchor": args.headroom,
        "per_shape": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--headroom", type=float, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--correctness-seeds", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--json-out")
    parser.add_argument("--reward-out")
    args = parser.parse_args()
    try:
        result = score(args)
    except Exception as exc:
        result = {
            "reward": 0.0,
            "correct": False,
            "reason": f"verifier error: {type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-1600:],
        }
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    print(payload)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(payload + "\n")
    if args.reward_out:
        Path(args.reward_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.reward_out).write_text(str(result.get("reward", 0.0)))


if __name__ == "__main__":
    main()
