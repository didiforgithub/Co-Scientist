        # K3_gdn2_longseq

        Optimize `/app/candidate.py` for a single NVIDIA H100. It must define
        `class ModelNew` with the same initialization and forward contract as
        `/app/reference_model.py`.

        ## Background

        K3 task: Gated DeltaNet 2 (GDN-2) — chunkwise delta rule with channel-wise gates.

GDN-2 generalizes the gated delta rule with TWO channel-wise gates: an erase gate
`b` on the key axis and a write gate `w` on the value axis (plus per-channel log
decay `g`). Collapsing b = w = scalar recovers KDA. There is NO native torch op —
the chunkwise WY-representation scan is a genuine hand-written kernel.

Math (per head, on matrix state S in R^{K x V}, verified against
fla.ops.gdn2.naive.naive_recurrent_gdn2):
    scale = 1/sqrt(K),  q <- q * scale
    for t:
        S      <- Diag(exp(g_t)) S              # per-channel decay on K axis
        erase  <- ((b_t * k_t) · S)             # gated read at key  -> R^V
        v_new  <- w_t * v_t - erase             # gated write minus erase
        S      <- S + k_t ⊗ v_new               # rank-1 update
        o_t    <- (q_t · S)                     # readout            -> R^V
    equivalently S_t = (I - k_t (b_t*k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t*v_t)^T

`g` = channel-wise log-decay [.,.,H,K]; `b` = erase gate [.,.,H,K]; `w` = write
gate [.,.,H,V]. This Model.forward IS the naive O(T) recurrent oracle.

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

        Final verification uses hidden random tensor values for every public shape. It checks the
        selected forward output only; backward gradients are not part of this source suite's verifier.

        ## Scoring

        The score is the geometric mean of `naive_latency / candidate_latency` over all shapes.
        Correctness failure on any shape gives zero reward. The naive anchor is `1.0`; the calibrated
        `fla.gdn2:chunk_gdn2` anchor is `218.1`. Reward is:

        ```text
        clip(0.5 * log(score) / log(218.1), 0, 1)
        ```

        ## Rules

        - Preserve `ModelNew` and support every listed shape.
        - Do not modify `reference_model.py`, `scorer.py`, or `evaluate.py`.
        - Do not inspect hidden verifier state or hard-code outputs.
        - Do not use the network or additional GPUs.
