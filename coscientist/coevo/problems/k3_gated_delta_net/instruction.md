        # K3_gated_delta_net

        Optimize `/app/candidate.py` for a single NVIDIA H100. It must define
        `class ModelNew` with the same initialization and forward contract as
        `/app/reference_model.py`.

        ## Background

        K3 task: Gated DeltaNet (GDN) — chunkwise linear attention with recurrent state.

REAL production hotspot: GDN layers are 75% of the layers in our Qwen3.6-27B /
3.5-35B-A3B models. Training goes through transformers' Qwen3NextGatedDeltaNet,
which calls the `fla` (flash-linear-attention) Triton kernels. There is NO native
torch op for this — the chunkwise-scan + recurrent-state update must be written by
hand, which is exactly why it is a genuine kernel-research task (unlike attention
or fused-loss, where cuBLAS/torch is already near-optimal).

Math (per v-head, gated delta rule, verified against transformers'
`torch_recurrent_gated_delta_rule`, modeling_qwen3_next.py:454):
    q, k <- L2norm(q), L2norm(k)          # optional, on by default in Qwen3Next
    q <- q * scale,  scale = 1/sqrt(K)
    for t in 0..T:
        S <- S * exp(g_t)                 # state decay, g_t log-space scalar per head
        delta <- (v_t - Sᵀ k_t) * beta_t  # delta rule correction
        S <- S + k_t ⊗ delta              # rank-1 state update
        o_t <- Sᵀ q_t                      # readout
    S: recurrent state [K, V] per (batch, v-head)

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

        - `[1, 128, 4, 8, 64, 64]`
- `[2, 256, 8, 16, 128, 128]`
- `[1, 512, 16, 48, 128, 128]`
- `[2, 512, 16, 32, 128, 128]`

        Final verification uses hidden random tensor values for every public shape. It checks the
        selected forward output only; backward gradients are not part of this source suite's verifier.

        ## Scoring

        The score is the geometric mean of `naive_latency / candidate_latency` over all shapes.
        Correctness failure on any shape gives zero reward. The naive anchor is `1.0`; the calibrated
        `fla.gated_delta_rule:chunk_gated_delta_rule` anchor is `46`. Reward is:

        ```text
        clip(0.5 * log(score) / log(46), 0, 1)
        ```

        ## Rules

        - Preserve `ModelNew` and support every listed shape.
        - Do not modify `reference_model.py`, `scorer.py`, or `evaluate.py`.
        - Do not inspect hidden verifier state or hard-code outputs.
        - Do not use the network or additional GPUs.
