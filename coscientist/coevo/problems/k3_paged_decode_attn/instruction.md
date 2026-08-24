        # K3_paged_decode_attn

        Optimize `/app/candidate.py` for a single NVIDIA H100. It must define
        `class ModelNew` with the same initialization and forward contract as
        `/app/reference_model.py`.

        ## Background

        K3 task: Paged KV-cache decode attention (the LLM inference decode hotspot).

REAL production hotspot: during autoregressive decoding, every new token attends
over the entire cached K/V history, which is stored in a *paged* KV cache (vLLM /
PagedAttention style) — physically non-contiguous 16-token pages indexed by a
per-sequence page table. This single-query-token attention over long paged history
is the dominant cost of LLM serving.

This Model.forward IS the naive reference: it gathers each sequence's pages back
into a contiguous [seq_len, H_kv, D] tensor, expands KV heads to query heads (GQA),
and does a plain softmax(q·kᵀ/√d)·v in fp32. Correct, but slow (python loop over
batch + full materialized scores). It is the CORRECTNESS oracle.

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
        `flashinfer` anchor is `13.2`. Reward is:

        ```text
        clip(0.5 * log(score) / log(13.2), 0, 1)
        ```

        ## Rules

        - Preserve `ModelNew` and support every listed shape.
        - Do not modify `reference_model.py`, `scorer.py`, or `evaluate.py`.
        - Do not inspect hidden verifier state or hard-code outputs.
        - Do not use the network or additional GPUs.
