        # K3_dplr_bigdim

        Optimize `/app/candidate.py` for a single NVIDIA H100. It must define
        `class ModelNew` with the same initialization and forward contract as
        `/app/reference_model.py`.

        ## Background

        K3 task: DPLR (bigdim) — Diagonal-Plus-Low-Rank delta rule, K=V=256 head dim.

DPLR (fla.ops.generalized_delta_rule.dplr) generalizes the delta rule: instead
of a scalar/diagonal state decay, the recurrent state is transformed each step by
a diagonal-plus-rank-1 matrix  Diag(exp(gk)) + a bᵀ  before the k⊗v write:
    S_t = Diag(exp(gk_t)) S_{t-1} + a_t (b_tᵀ S_{t-1}) + k_t ⊗ v_t
        = exp(gk_t) ⊙ S_{t-1} + a_t ⊗ (S_{t-1}ᵀ a_t? )  ... (see exact form below)
    o_t = q_tᵀ S_t
This is the RWKV-7 / generalized-delta family. There is NO native torch op for
the chunkwise DPLR scan — it must be written by hand, which is what makes it a
genuine kernel-research task.

Exact per-step recurrence (verified against fla.ops...dplr.naive.dplr_recurrence
and the fused_recurrent kernel):
    q <- q * scale,  scale = 1/sqrt(K)
    for t in 0..T:
        kv = k_t ⊗ v_t + b_t ⊗ (a_tᵀ S_{t-1})      # low-rank correction via a,b
        S  = exp(gk_t)[:, None] ⊙ S_{t-1} + kv     # diagonal decay + write
        o_t = q_tᵀ S
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

        - `[1, 128, 4, 256, 256]`
- `[2, 256, 4, 256, 256]`
- `[1, 512, 4, 256, 256]`
- `[2, 256, 8, 256, 256]`

        Final verification uses hidden random tensor values for every public shape. It checks the
        selected forward output only; backward gradients are not part of this source suite's verifier.

        ## Scoring

        The score is the geometric mean of `naive_latency / candidate_latency` over all shapes.
        Correctness failure on any shape gives zero reward. The naive anchor is `1.0`; the calibrated
        `fla.generalized_delta_rule.dplr:chunk_dplr_delta_rule` anchor is `49.5`. Reward is:

        ```text
        clip(0.5 * log(score) / log(49.5), 0, 1)
        ```

        ## Rules

        - Preserve `ModelNew` and support every listed shape.
        - Do not modify `reference_model.py`, `scorer.py`, or `evaluate.py`.
        - Do not inspect hidden verifier state or hard-code outputs.
        - Do not use the network or additional GPUs.
