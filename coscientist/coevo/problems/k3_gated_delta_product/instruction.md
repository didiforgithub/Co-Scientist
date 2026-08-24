        # K3_gated_delta_product

        Optimize `/app/candidate.py` for a single NVIDIA H100. It must define
        `class ModelNew` with the same initialization and forward contract as
        `/app/reference_model.py`.

        ## Background

        K3 task: Gated DeltaProduct — multi-householder gated delta rule.

DeltaProduct applies MULTIPLE (num_householder) delta-rule state transitions per
time step, i.e. a product of Householder-like rank-1 updates, giving a richer
per-token state transition than a single delta update. There is NO native torch
op — the chunkwise scan over the expanded (T * num_householder) key/value stream
is a genuine hand-written kernel.

Math (per head, on matrix state h in R^{K x V}, verified against
fla.ops.gated_delta_product.naive.naive_recurrent_gated_delta_product):
    scale = 1/sqrt(K),  q <- q * scale
    for t in 0..T:
        h <- exp(g_t) * h                                  # head-wise log decay
        for j in 0..num_householder:                       # product of updates
            idx = t*num_householder + j
            h <- h + (v_idx - (k_idx · h)) * beta_idx  ⊗ k_idx   # delta update
        o_t <- (q_t · h)                                   # readout
    h: R^{K x V} recurrent state per (batch, head)

`g` is the head-wise log-decay applied ONCE per real time step [B,T,H]. `k`, `v`,
`beta` live on the expanded axis of length T*num_householder. This Model.forward IS
the naive O(T * num_householder) recurrent oracle.

NOTE: fla's chunk kernel requires bf16 inputs (float32 unsupported), so the
workload uses bfloat16; allclose tolerance is the standard 1e-2.

        ## Iteration loop

        - Edit `/app/candidate.py`; PyTorch, Triton, CUDA headers, `nvcc`, and Ninja are available.
        - Run `python /app/evaluate.py` for a one-shape quick check.
        - Run `python /app/evaluate.py --full` for the complete public workload.
        - Repeat for the full two-hour Agent budget.
        - The FLA or FlashInfer baseline named in `reference_model.baseline()` is documentation and
          calibration metadata; those external packages are not installed in the task image.

        ## Workload

        - `[1, 128, 4, 64, 64, 2]`
- `[2, 256, 4, 128, 128, 2]`
- `[1, 512, 8, 128, 128, 2]`
- `[2, 256, 8, 64, 64, 2]`

        Final verification uses hidden random tensor values for every public shape. It checks the
        selected forward output only; backward gradients are not part of this source suite's verifier.

        ## Scoring

        The score is the geometric mean of `naive_latency / candidate_latency` over all shapes.
        Correctness failure on any shape gives zero reward. The naive anchor is `1.0`; the calibrated
        `fla.gated_delta_product:chunk_gated_delta_product` anchor is `51.3`. Reward is:

        ```text
        clip(0.5 * log(score) / log(51.3), 0, 1)
        ```

        ## Rules

        - Preserve `ModelNew` and support every listed shape.
        - Do not modify `reference_model.py`, `scorer.py`, or `evaluate.py`.
        - Do not inspect hidden verifier state or hard-code outputs.
        - Do not use the network or additional GPUs.
