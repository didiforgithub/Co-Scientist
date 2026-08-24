        # K3_paged_decode_fp8

        Optimize `/app/candidate.py` for a single NVIDIA H100. It must define
        `class ModelNew` with the same initialization and forward contract as
        `/app/reference_model.py`.

        ## Background

        K3 task: Paged KV-cache decode attention with an FP8 (e4m3) KV cache.

REAL production hotspot: to halve KV-cache memory bandwidth and capacity during
long-context LLM serving, the paged KV cache is stored in FP8 (e4m3). The decode
kernel reads the fp8 pages, dequantizes on the fly, and does single-query-token
attention over the whole cached history (GQA). Query stays in fp16. This is the
fp8-KV variant of the PagedAttention decode hotspot (vLLM / TRT-LLM fp8 KV).

This Model.forward IS the naive reference: it gathers each sequence's fp8 pages,
upcasts (dequantizes) them to fp32, expands KV heads to query heads (GQA), and does
a plain softmax(q·kᵀ/√d)·v in fp32. Correct, but slow. It is the CORRECTNESS oracle.

        ## Iteration loop

        - Edit `/app/candidate.py`; PyTorch, Triton, CUDA headers, `nvcc`, and Ninja are available.
        - Run `python /app/evaluate.py` for a one-shape quick check.
        - Run `python /app/evaluate.py --full` for the complete public workload.
        - Repeat for the full two-hour Agent budget.
        - The FLA or FlashInfer baseline named in `reference_model.baseline()` is documentation and
          calibration metadata; those external packages are not installed in the task image.

        ## Workload

        - `[4, 32, 8, 128, 256, 16]`
- `[8, 32, 8, 128, 512, 16]`
- `[16, 32, 8, 128, 1024, 16]`
- `[32, 64, 8, 128, 2048, 16]`

        Final verification uses hidden random tensor values for every public shape. It checks the
        selected forward output only; backward gradients are not part of this source suite's verifier.

        ## Scoring

        The score is the geometric mean of `naive_latency / candidate_latency` over all shapes.
        Correctness failure on any shape gives zero reward. The naive anchor is `1.0`; the calibrated
        `flashinfer` anchor is `11.8`. Reward is:

        ```text
        clip(0.5 * log(score) / log(11.8), 0, 1)
        ```

        ## Rules

        - Preserve `ModelNew` and support every listed shape.
        - Do not modify `reference_model.py`, `scorer.py`, or `evaluate.py`.
        - Do not inspect hidden verifier state or hard-code outputs.
        - Do not use the network or additional GPUs.
