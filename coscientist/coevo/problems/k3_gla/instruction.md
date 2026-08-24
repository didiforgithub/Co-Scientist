        # K3_gla

        Optimize `/app/candidate.py` for a single NVIDIA H100. It must define
        `class ModelNew` with the same initialization and forward contract as
        `/app/reference_model.py`.

        ## Background

        K3 task: GLA (Gated Linear Attention) — chunkwise, per-channel forget gate.

Linear attention where every KEY channel has its own data-dependent forget
gate (log-space). No delta correction — just gated accumulation of k⊗v.
No native torch op: the chunkwise-scan with a per-channel decay must be
written by hand, which is what makes it a genuine kernel-research task.

Math (per head, verified against fla.ops.gla.naive.naive_recurrent_gla):
    q <- q * scale,  scale = 1/sqrt(K)
    for t in 0..T:
        S   <- S * exp(g_t)[:, None] + k_t ⊗ v_t   # g_t decays each K channel
        o_t <- qᵀ_t S                              # readout
    g_t: log-space per-channel gate of shape [K]
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
        `fla.gla:chunk_gla` anchor is `56.3`. Reward is:

        ```text
        clip(0.5 * log(score) / log(56.3), 0, 1)
        ```

        ## Rules

        - Preserve `ModelNew` and support every listed shape.
        - Do not modify `reference_model.py`, `scorer.py`, or `evaluate.py`.
        - Do not inspect hidden verifier state or hard-code outputs.
        - Do not use the network or additional GPUs.
