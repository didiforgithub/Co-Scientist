        # K3_based

        Optimize `/app/candidate.py` for a single NVIDIA H100. It must define
        `class ModelNew` with the same initialization and forward contract as
        `/app/reference_model.py`.

        ## Background

        K3 task: Based — linear attention with 2nd-order Taylor feature map.

Based (Zoology / "Simple linear attention language models", arXiv:2402.18668)
approximates softmax attention with the 2nd-order Taylor expansion of exp:
    exp(qᵀk) ≈ 1 + qᵀk + (qᵀk)²/2
This is a linear-attention operator: the feature map phi(x) turns attention
into a running sum of outer products of feature vectors, so it can be evaluated
with a causal recurrent state instead of the O(T²) attention matrix. There is no
native torch op for the chunkwise Taylor-feature scan — it must be written by
hand, which is what makes it a genuine kernel-research task.

Math (per head, verified against fla.ops.based.naive.naive_parallel_based):
    q <- q * scale,  scale = 1/sqrt(K)
    A_{ts} = 1 + qᵀ_t k_s + 0.5 (qᵀ_t k_s)²     for s <= t   (causal, inclusive)
    o_t = (sum_{s<=t} A_{ts} v_s) / (sum_{s<=t} A_{ts} + eps)   # use_norm=True
The numerator/denominator are accumulated with running Taylor moments:
    S0 = sum v_s               (zero-order, coeff 1)
    S1 = sum k_s ⊗ v_s         (first-order)
    S2 = sum (k_s⊗k_s) ⊗ v_s   (second-order, coeff 1/2)
and the matching key moments Z1, Z2 for the denominator.

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

        - `[1, 128, 4, 16, 64]`
- `[2, 256, 4, 16, 64]`
- `[1, 512, 8, 16, 128]`
- `[2, 512, 8, 16, 128]`

        Final verification uses hidden random tensor values for every public shape. It checks the
        selected forward output only; backward gradients are not part of this source suite's verifier.

        ## Scoring

        The score is the geometric mean of `naive_latency / candidate_latency` over all shapes.
        Correctness failure on any shape gives zero reward. The naive anchor is `1.0`; the calibrated
        `fla.based:fused_chunk_based` anchor is `124.5`. Reward is:

        ```text
        clip(0.5 * log(score) / log(124.5), 0, 1)
        ```

        ## Rules

        - Preserve `ModelNew` and support every listed shape.
        - Do not modify `reference_model.py`, `scorer.py`, or `evaluate.py`.
        - Do not inspect hidden verifier state or hard-code outputs.
        - Do not use the network or additional GPUs.
