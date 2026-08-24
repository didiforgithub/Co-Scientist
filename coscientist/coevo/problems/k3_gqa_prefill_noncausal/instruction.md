        # K3_gqa_prefill_noncausal

        Optimize `/app/candidate.py` for a single NVIDIA H100. It must define
        `class ModelNew` with the same initialization and forward contract as
        `/app/reference_model.py`.

        ## Background

        K3 task: Ragged (variable-length) NON-causal bidirectional prefill with GQA.

REAL production hotspot: encoder-style / bidirectional attention (embedding models,
prefix-LM, cross-attention encoders) packs many variable-length sequences into one
flat buffer (cu_seqlens) and every token attends to all tokens of its own sequence
(no causal mask). Heavy GQA (group = 8) is common. This is the varlen full-attention
path — the counterpart to the causal ragged prefill.

This Model.forward IS the naive reference: per-sequence full (non-causal) softmax
attention with KV heads repeat-interleaved to query heads, in fp32. Correct, slow.

        ## Iteration loop

        - Edit `/app/candidate.py`; PyTorch, Triton, CUDA headers, `nvcc`, and Ninja are available.
        - Run `python /app/evaluate.py` for a one-shape quick check.
        - Run `python /app/evaluate.py --full` for the complete public workload.
        - Repeat for the full two-hour Agent budget.
        - The FLA or FlashInfer baseline named in `reference_model.baseline()` is documentation and
          calibration metadata; those external packages are not installed in the task image.

        ## Workload

        - `[[64, 128, 96, 200], 64, 8, 128]`
- `[[256, 384, 128, 512], 64, 8, 128]`
- `[[512, 768, 256, 1024, 640], 64, 8, 128]`
- `[[1024, 1536, 768, 2048, 512, 900], 64, 8, 128]`

        Final verification uses hidden random tensor values for every public shape. It checks the
        selected forward output only; backward gradients are not part of this source suite's verifier.

        ## Scoring

        The score is the geometric mean of `naive_latency / candidate_latency` over all shapes.
        Correctness failure on any shape gives zero reward. The naive anchor is `1.0`; the calibrated
        `flashinfer` anchor is `3`. Reward is:

        ```text
        clip(0.5 * log(score) / log(3), 0, 1)
        ```

        ## Rules

        - Preserve `ModelNew` and support every listed shape.
        - Do not modify `reference_model.py`, `scorer.py`, or `evaluate.py`.
        - Do not inspect hidden verifier state or hard-code outputs.
        - Do not use the network or additional GPUs.
