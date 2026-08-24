        # K3_gsa_longseq

        Optimize `/app/candidate.py` for a single NVIDIA H100. It must define
        `class ModelNew` with the same initialization and forward contract as
        `/app/reference_model.py`.

        ## Background

        K3 task: Gated Slot Attention (GSA) — chunkwise linear attention with a slot bottleneck.

GSA routes information through a small set of `M` memory slots via two stacked
gated linear-attention recurrences (an ABC-style bottleneck). There is NO native
torch op — the two chunkwise scans over the slot dimension must be written by hand.

Math (per head, verified against fla.ops.gsa.naive.naive_recurrent_gsa):
    scale = 1/sqrt(K),  q <- q * scale
    # stage 1 — write real keys into slot memory hk in R^{K x M}, read with q:
    for t: hk <- hk * exp(g_t)[None,:] + k_t ⊗ s_t ;  ok_t = (q_t · hk)   # -> R^M
    qv = softmax(ok, dim=slots)                                            # slot attention
    # stage 2 — write real values into slot memory hv in R^{M x V}, read with qv:
    for t: hv <- hv * exp(g_t)[:,None] + s_t ⊗ v_t ;  o_t = (qv_t · hv)    # -> R^V
    hk: R^{K x M}, hv: R^{M x V} recurrent states per (batch, head)

`g` is the log-space forget gate on the M slots. `s` is the slot representation
(acts as value in stage 1 and as key in stage 2). This Model.forward IS the naive
O(T) recurrent oracle: correct, differentiable, slow.

        ## Iteration loop

        - Edit `/app/candidate.py`; PyTorch, Triton, CUDA headers, `nvcc`, and Ninja are available.
        - Run `python /app/evaluate.py` for a one-shape quick check.
        - Run `python /app/evaluate.py --full` for the complete public workload.
        - Repeat for the full two-hour Agent budget.
        - The FLA or FlashInfer baseline named in `reference_model.baseline()` is documentation and
          calibration metadata; those external packages are not installed in the task image.

        ## Workload

        - `[1, 1024, 8, 128, 128, 64]`
- `[1, 2048, 8, 128, 128, 64]`
- `[1, 4096, 8, 128, 128, 64]`

        Final verification uses hidden random tensor values for every public shape. It checks the
        selected forward output only; backward gradients are not part of this source suite's verifier.

        ## Scoring

        The score is the geometric mean of `naive_latency / candidate_latency` over all shapes.
        Correctness failure on any shape gives zero reward. The naive anchor is `1.0`; the calibrated
        `fla.gsa:chunk_gsa` anchor is `428.6`. Reward is:

        ```text
        clip(0.5 * log(score) / log(428.6), 0, 1)
        ```

        ## Rules

        - Preserve `ModelNew` and support every listed shape.
        - Do not modify `reference_model.py`, `scorer.py`, or `evaluate.py`.
        - Do not inspect hidden verifier state or hard-code outputs.
        - Do not use the network or additional GPUs.
