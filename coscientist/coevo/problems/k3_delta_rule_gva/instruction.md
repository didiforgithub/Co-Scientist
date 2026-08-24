        # K3_delta_rule_gva

        Optimize `/app/candidate.py` for a single NVIDIA H100. It must define
        `class ModelNew` with the same initialization and forward contract as
        `/app/reference_model.py`.

        ## Background

        K3 task: DeltaNet (delta rule), GVA variant — grouped value attention.

Same error-driven delta-rule state update as K3_delta_rule (no decay gate), but
with GROUPED VALUE ATTENTION: there are H query/key heads and HV = 4*H value
heads. Each q/k head is shared (repeat_interleaved) across `groups = HV/H` value
heads, so the state is [HV, K, V]. Under GVA the head-sharing changes the memory
/ occupancy tradeoff, making it a distinct kernel-research target from the plain
per-head delta rule. There is no native torch op — the chunkwise-scan + recurrent
delta correction must be written by hand.

Math (per v-head, verified against fla.ops.gated_delta_rule.chunk_gated_delta_rule
with g == 0, i.e. no decay == pure delta rule; that op supports GVA whereas
chunk_delta_rule requires H == HV):
    q, k <- L2norm(q), L2norm(k)   # optional, matches use_qk_l2norm_in_kernel
    q    <- q * scale,  scale = 1/sqrt(K)
    (expand q,k from H heads to HV heads via repeat_interleave)
    for t in 0..T:
        delta <- (v_t - Sᵀ k_t) * beta_t   # error-driven correction
        S     <- S + k_t ⊗ delta           # rank-1 state update (no decay)
        o_t   <- Sᵀ q_t                     # readout
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

        - `[1, 128, 4, 16, 64, 64]`
- `[2, 256, 8, 32, 128, 128]`
- `[1, 512, 8, 32, 128, 128]`
- `[2, 512, 4, 16, 128, 128]`

        Final verification uses hidden random tensor values for every public shape. It checks the
        selected forward output only; backward gradients are not part of this source suite's verifier.

        ## Scoring

        The score is the geometric mean of `naive_latency / candidate_latency` over all shapes.
        Correctness failure on any shape gives zero reward. The naive anchor is `1.0`; the calibrated
        `fla.gated_delta_rule:chunk_gated_delta_rule` anchor is `36.7`. Reward is:

        ```text
        clip(0.5 * log(score) / log(36.7), 0, 1)
        ```

        ## Rules

        - Preserve `ModelNew` and support every listed shape.
        - Do not modify `reference_model.py`, `scorer.py`, or `evaluate.py`.
        - Do not inspect hidden verifier state or hard-code outputs.
        - Do not use the network or additional GPUs.
