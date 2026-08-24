        # K3_append_paged_attn

        Optimize `/app/candidate.py` for a single NVIDIA H100. It must define
        `class ModelNew` with the same initialization and forward contract as
        `/app/reference_model.py`.

        ## Background

        K3 task: Append / chunked-prefill attention over a paged KV cache.

REAL production hotspot: continuous-batching LLM serving (vLLM) does *append*
attention — each request already has a KV cache of `kv_len` tokens stored in a
paged cache, and a chunk of `q_len` NEW query tokens is appended this step. Each
new query token attends over the whole cache (prefix + the new tokens up to and
including itself), with causal masking anchored at its absolute position
(kv_len - q_len + i). This is the unifying kernel behind both prefill (q_len ==
kv_len) and speculative/chunked decode (q_len << kv_len).

This Model.forward IS the naive reference: it gathers each request's KV pages
into a contiguous [kv_len, Hkv, D] tensor, expands KV heads to query heads (GQA),
and does a plain position-anchored causal softmax(q·kᵀ/√d)·v in fp32, per request.
Correct, but slow (python loop over batch + full materialized scores). It is the
CORRECTNESS oracle.

        ## Iteration loop

        - Edit `/app/candidate.py`; PyTorch, Triton, CUDA headers, `nvcc`, and Ninja are available.
        - Run `python /app/evaluate.py` for a one-shape quick check.
        - Run `python /app/evaluate.py --full` for the complete public workload.
        - Repeat for the full two-hour Agent budget.
        - The FLA or FlashInfer baseline named in `reference_model.baseline()` is documentation and
          calibration metadata; those external packages are not installed in the task image.

        ## Workload

        - `[[[16, 256], [32, 512]], 32, 8, 128, 16]`
- `[[[32, 512], [64, 1024], [16, 256]], 32, 8, 128, 16]`
- `[[[64, 1024], [128, 2048], [32, 512], [16, 256]], 32, 8, 128, 16]`
- `[[[256, 2048], [128, 4096], [64, 1024]], 40, 8, 128, 16]`

        Final verification uses hidden random tensor values for every public shape. It checks the
        selected forward output only; backward gradients are not part of this source suite's verifier.

        ## Scoring

        The score is the geometric mean of `naive_latency / candidate_latency` over all shapes.
        Correctness failure on any shape gives zero reward. The naive anchor is `1.0`; the calibrated
        `flashinfer` anchor is `3.5`. Reward is:

        ```text
        clip(0.5 * log(score) / log(3.5), 0, 1)
        ```

        ## Rules

        - Preserve `ModelNew` and support every listed shape.
        - Do not modify `reference_model.py`, `scorer.py`, or `evaluate.py`.
        - Do not inspect hidden verifier state or hard-code outputs.
        - Do not use the network or additional GPUs.
