        # K3_delta_rule_longseq

        Optimize `/app/candidate.py` for a single NVIDIA H100. It must define
        `class ModelNew` with the same initialization and forward contract as
        `/app/reference_model.py`.

        ## Background

        K3 task: DeltaNet (delta rule) — chunkwise linear attention, no gate.

Linear-attention variant with a *delta rule* state update but NO decay gate:
the state is corrected toward each new value by an error-driven, beta-scaled
rank-1 update. There is no native torch op — the chunkwise-scan + recurrent
delta correction must be written by hand, which is what makes it a genuine
kernel-research task.

Math (per head, verified against fla.ops.delta_rule.naive.delta_rule_recurrence):
    q, k <- L2norm(q), L2norm(k)   # optional, matches use_qk_l2norm_in_kernel
    q    <- q * scale,  scale = 1/sqrt(K)
    for t in 0..T:
        delta <- (v_t - Sᵀ k_t) * beta_t   # error-driven correction
        S     <- S + k_t ⊗ delta           # rank-1 state update (no decay)
        o_t   <- Sᵀ q_t                     # readout
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

        - `[1, 1024, 8, 128, 128]`
- `[1, 2048, 8, 128, 128]`
- `[1, 4096, 8, 128, 128]`

        Final verification uses hidden random tensor values for every public shape. It checks the
        selected forward output only; backward gradients are not part of this source suite's verifier.

        ## Scoring

        The score is the geometric mean of `naive_latency / candidate_latency` over all shapes.
        Correctness failure on any shape gives zero reward. The naive anchor is `1.0`; the calibrated
        `fla.delta_rule:chunk_delta_rule` anchor is `250`. Reward is:

        ```text
        clip(0.5 * log(score) / log(250), 0, 1)
        ```

        ## Rules

        - Preserve `ModelNew` and support every listed shape.
        - Do not modify `reference_model.py`, `scorer.py`, or `evaluate.py`.
        - Do not inspect hidden verifier state or hard-code outputs.
        - Do not use the network or additional GPUs.
