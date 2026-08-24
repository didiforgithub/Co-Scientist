        # K3_abc_bf16

        Optimize `/app/candidate.py` for a single NVIDIA H100. It must define
        `class ModelNew` with the same initialization and forward contract as
        `/app/reference_model.py`.

        ## Background

        K3 task: ABC (Attention with Bounded-memory Control) — slot-memory linear attn.

ABC routes information through a small set of `M` memory slots via two stacked
linear-attention recurrences. Unlike GSA, ABC has NO separately supplied forget
gate: the slot gate `g` and the normalized slot weights are *derived from the raw
slot scores `s`* via a log-cumsum-exp / softmax-in-time trick. There is NO native
torch op — the two chunkwise scans plus the in-kernel gate derivation must be
written by hand.

Math (per head, verified against fla.ops.abc.naive.naive_recurrent_abc, g=None path):
    scale = 1/sqrt(K)
    # derive slot gate & normalized slot weights from raw scores s (over time):
    z   = logcumsumexp(s, dim=T)                         # running log-normalizer
    g   = cat(z[:1], z[:-1]) - z    (shifted)            # log slot forget gate
    s'  = exp(s - z)                                     # time-normalized slots
    # stage 1 — write real keys into slot memory hk in R^{K x M}, read with q:
    for t: hk <- hk*exp(g_t)[None,:] + k_t ⊗ s'_t ;  ok_t = q_t·hk    # -> R^M
    qv = softmax(ok, dim=slots)
    # stage 2 — write real values into slot memory hv in R^{M x V}, read with qv:
    for t: hv <- hv*exp(g_t)[:,None] + s'_t ⊗ v_t ;  o_t = qv_t·hv    # -> R^V
    hk: R^{K x M}, hv: R^{M x V} recurrent states per (batch, head)

This Model.forward IS the naive O(T) recurrent oracle: correct, differentiable, slow.

        ## Iteration loop

        - Edit `/app/candidate.py`; PyTorch, Triton, CUDA headers, `nvcc`, and Ninja are available.
        - Run `python /app/evaluate.py` for a one-shape quick check.
        - Run `python /app/evaluate.py --full` for the complete public workload.
        - Repeat for the full two-hour Agent budget.
        - The FLA or FlashInfer baseline named in `reference_model.baseline()` is documentation and
          calibration metadata; those external packages are not installed in the task image.

        ## Workload

        - `[1, 128, 4, 64, 64, 32]`
- `[2, 256, 4, 64, 64, 32]`
- `[1, 512, 8, 64, 64, 32]`
- `[2, 256, 8, 64, 64, 32]`

        Final verification uses hidden random tensor values for every public shape. It checks the
        selected forward output only; backward gradients are not part of this source suite's verifier.

        ## Scoring

        The score is the geometric mean of `naive_latency / candidate_latency` over all shapes.
        Correctness failure on any shape gives zero reward. The naive anchor is `1.0`; the calibrated
        `fla.abc:chunk_abc` anchor is `77.8`. Reward is:

        ```text
        clip(0.5 * log(score) / log(77.8), 0, 1)
        ```

        ## Rules

        - Preserve `ModelNew` and support every listed shape.
        - Do not modify `reference_model.py`, `scorer.py`, or `evaluate.py`.
        - Do not inspect hidden verifier state or hard-code outputs.
        - Do not use the network or additional GPUs.
