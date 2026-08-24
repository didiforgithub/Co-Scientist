        # K3_rwkv7_bigdim

        Optimize `/app/candidate.py` for a single NVIDIA H100. It must define
        `class ModelNew` with the same initialization and forward contract as
        `/app/reference_model.py`.

        ## Background

        K3 task: RWKV-7 — DPLR (diagonal-plus-low-rank) delta-rule linear attention.

RWKV-7 ("Goose") generalizes the delta rule to a diagonal-plus-low-rank state
transition: each step the state is decayed per-channel (diagonal `w`) AND
corrected by a rank-1 term built from two learned low-rank vectors `a`,`b`
(the "in-context learning" removal/replacement keys). There is NO native torch
op — the chunkwise DPLR scan (fla routes chunk_rwkv7 -> chunk_dplr_delta_rule)
must be hand-written.

Math (per head, verified against
fla.ops.generalized_delta_rule.dplr.naive.dplr_recurrence, scale=1.0):
    for t in 0..T:
        kv  <- k_t ⊗ v_t + ((S · a_t) summed over K) ⊗ b_t   # rank-1 DPLR correction
        S   <- S * exp(w_t)[:,None] + kv                      # per-K-channel diag decay
        o_t <- rᵀ_t S                                          # readout (r = query)
    w_t: log-space per-channel decay [K];  a_t,b_t: low-rank vectors [K]
    S: recurrent state [K, V] per (batch, head)

This Model.forward IS the naive O(T) recurrent oracle: correct, differentiable, slow.

        ## Iteration loop

        - Edit `/app/candidate.py`; PyTorch, Triton, CUDA headers, `nvcc`, and Ninja are available.
        - Run `python /app/evaluate.py` for a one-shape quick check.
        - Run `python /app/evaluate.py --full` for the complete public workload.
        - Repeat for the full two-hour Agent budget.
        - The FLA or FlashInfer baseline named in `reference_model.baseline()` is documentation and
          calibration metadata; those external packages are not installed in the task image.

        ## Workload

        - `[1, 256, 4, 256, 256]`
- `[2, 256, 8, 256, 256]`
- `[1, 512, 4, 256, 256]`

        Final verification uses hidden random tensor values for every public shape. It checks the
        selected forward output only; backward gradients are not part of this source suite's verifier.

        ## Scoring

        The score is the geometric mean of `naive_latency / candidate_latency` over all shapes.
        Correctness failure on any shape gives zero reward. The naive anchor is `1.0`; the calibrated
        `fla.rwkv7:chunk_rwkv7` anchor is `103.9`. Reward is:

        ```text
        clip(0.5 * log(score) / log(103.9), 0, 1)
        ```

        ## Rules

        - Preserve `ModelNew` and support every listed shape.
        - Do not modify `reference_model.py`, `scorer.py`, or `evaluate.py`.
        - Do not inspect hidden verifier state or hard-code outputs.
        - Do not use the network or additional GPUs.
