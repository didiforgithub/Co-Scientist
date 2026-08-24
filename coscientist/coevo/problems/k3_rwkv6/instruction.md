        # K3_rwkv6

        Optimize `/app/candidate.py` for a single NVIDIA H100. It must define
        `class ModelNew` with the same initialization and forward contract as
        `/app/reference_model.py`.

        ## Background

        K3 task: RWKV-6 — chunkwise linear attention with log-decay + bonus.

RWKV-6 receptance-weighted key-value: like GLA (per-channel log-space decay
`w` on the state) but with an extra "bonus" term `u` that boosts the CURRENT
timestep's contribution before it is folded into the decaying state. The
readout mixes the pre-update state with the u-boosted current k⊗v. No native
torch op: the chunkwise-scan with decay + bonus must be written by hand, which
is what makes it a genuine kernel-research task.

Math (per head, verified against fla.ops.rwkv6.recurrent_naive.naive_recurrent_rwkv6):
    r <- r * scale,  scale = 1/sqrt(K)          # r is the "query"/receptance
    for t in 0..T:
        kv_t <- k_t ⊗ v_t                        # (K, V)
        o_t  <- rᵀ_t (S + u ⊙ kv_t)              # bonus u boosts current step
        S    <- S * exp(w_t)[:, None] + kv_t     # per-channel log decay w_t
    w_t: log-space per-channel decay [K];  u: per-head bonus [K]
    S: recurrent state [K, V] per (batch, head)

This Model.forward IS the naive O(T) recurrent reference: correct and
differentiable, but slow (a Python loop over T). It is the CORRECTNESS oracle.

        ## Iteration loop

        - Edit `/app/candidate.py`; PyTorch, Triton, CUDA headers, `nvcc`, and Ninja are available.
        - Run `python /app/evaluate.py` for a one-shape quick check.
        - Run `python /app/evaluate.py --full` for the complete public workload.
        - Repeat for the full two-hour Agent budget.
        - The FLA or FlashInfer baseline named in `reference_model.baseline()` is documentation and
          calibration metadata; those external packages are not installed in the task image.

        ## Workload

        - `[1, 128, 4, 64, 64]`
- `[2, 256, 8, 128, 128]`
- `[1, 512, 8, 128, 128]`
- `[2, 512, 16, 128, 128]`

        Final verification uses hidden random tensor values for every public shape. It checks the
        selected forward output only; backward gradients are not part of this source suite's verifier.

        ## Scoring

        The score is the geometric mean of `naive_latency / candidate_latency` over all shapes.
        Correctness failure on any shape gives zero reward. The naive anchor is `1.0`; the calibrated
        `fla.rwkv6:chunk_rwkv6` anchor is `58`. Reward is:

        ```text
        clip(0.5 * log(score) / log(58), 0, 1)
        ```

        ## Rules

        - Preserve `ModelNew` and support every listed shape.
        - Do not modify `reference_model.py`, `scorer.py`, or `evaluate.py`.
        - Do not inspect hidden verifier state or hard-code outputs.
        - Do not use the network or additional GPUs.
