        # K3_comba_longseq

        Optimize `/app/candidate.py` for a single NVIDIA H100. It must define
        `class ModelNew` with the same initialization and forward contract as
        `/app/reference_model.py`.

        ## Background

        K3 task: COMBA — gated delta rule with an auxiliary key `p`.

COMBA is a delta-rule variant where the *erase/correction* read uses a separate
auxiliary key `p` (instead of reusing `k`), while the rank-1 write still uses `k`.
This decouples "what to forget" from "what to write". There is NO native torch op —
the chunkwise WY-representation scan is a genuine hand-written kernel.

Math (per head, on matrix state h in R^{K x V}, verified against
fla.ops.comba.naive.naive_recurrent_comba):
    scale = 1/sqrt(K),  q <- q * scale
    for t:
        h      <- exp(g_t) * h                  # scalar (head-wise) log decay
        v_new  <- (v_t - (p_t · h)) * beta_t    # delta correction read with p
        h      <- h + k_t ⊗ v_new               # rank-1 write with k
        o_t    <- (q_t · h)                      # readout
    h: R^{K x V} recurrent state per (batch, head)

`g` = head-wise log-decay [.,.,H]; `beta` = update gate [.,.,H]; `p` = auxiliary
key [.,.,H,K]. This Model.forward IS the naive O(T) recurrent oracle.

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
        `fla.comba:chunk_comba` anchor is `306.2`. Reward is:

        ```text
        clip(0.5 * log(score) / log(306.2), 0, 1)
        ```

        ## Rules

        - Preserve `ModelNew` and support every listed shape.
        - Do not modify `reference_model.py`, `scorer.py`, or `evaluate.py`.
        - Do not inspect hidden verifier state or hard-code outputs.
        - Do not use the network or additional GPUs.
