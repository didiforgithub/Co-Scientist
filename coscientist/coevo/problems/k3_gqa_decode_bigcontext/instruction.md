        # K3_gqa_decode_bigcontext

        Optimize `/app/candidate.py` for a single NVIDIA H100. It must define
        `class ModelNew` with the same initialization and forward contract as
        `/app/reference_model.py`.

        ## Background

        K3 task: Long-context paged GQA decode attention (memory-bound decode stress).

REAL production hotspot: at long context (many thousands of cached tokens) the
autoregressive decode step is fully memory-bandwidth bound — every new token must
read the entire paged KV history to produce one output token. Aggressive GQA
(group = 8, only 4 KV heads) is used to shrink that KV read. This is the long-context
variant of PagedAttention decode where the KV read dominates end-to-end latency.

This Model.forward IS the naive reference: it gathers each sequence's pages into a
contiguous [seq_len, Hkv, D] tensor, expands KV heads to query heads (GQA), and does
a plain softmax(q·kᵀ/√d)·v in fp32. Correct, but slow. It is the CORRECTNESS oracle.

        ## Iteration loop

        - Edit `/app/candidate.py`; PyTorch, Triton, CUDA headers, `nvcc`, and Ninja are available.
        - Run `python /app/evaluate.py` for a one-shape quick check.
        - Run `python /app/evaluate.py --full` for the complete public workload.
        - Repeat for the full two-hour Agent budget.
        - The FLA or FlashInfer baseline named in `reference_model.baseline()` is documentation and
          calibration metadata; those external packages are not installed in the task image.

        ## Workload

        - `[2, 32, 4, 128, 2048, 16]`
- `[4, 32, 4, 128, 4096, 16]`
- `[4, 32, 4, 128, 8192, 16]`
- `[8, 32, 4, 128, 8192, 16]`

        Final verification uses hidden random tensor values for every public shape. It checks the
        selected forward output only; backward gradients are not part of this source suite's verifier.

        ## Scoring

        The score is the geometric mean of `naive_latency / candidate_latency` over all shapes.
        Correctness failure on any shape gives zero reward. The naive anchor is `1.0`; the calibrated
        `flashinfer` anchor is `7.3`. Reward is:

        ```text
        clip(0.5 * log(score) / log(7.3), 0, 1)
        ```

        ## Rules

        - Preserve `ModelNew` and support every listed shape.
        - Do not modify `reference_model.py`, `scorer.py`, or `evaluate.py`.
        - Do not inspect hidden verifier state or hard-code outputs.
        - Do not use the network or additional GPUs.
