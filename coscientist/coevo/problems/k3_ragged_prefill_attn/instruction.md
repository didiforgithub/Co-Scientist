        # K3_ragged_prefill_attn

        Optimize `/app/candidate.py` for a single NVIDIA H100. It must define
        `class ModelNew` with the same initialization and forward contract as
        `/app/reference_model.py`.

        ## Background

        K3 task: Ragged (variable-length) causal prefill attention.

REAL production hotspot: at LLM prefill / training time, a batch packs many
prompts of *different* lengths into one flat token buffer described by cumulative
sequence-length offsets (cu_seqlens). Each prompt does full causal self-attention
over only its own tokens — no cross-prompt attention. This is the varlen FlashAttention
path (vLLM prefill, packed SFT batches).

This Model.forward IS the naive reference: it slices out each sequence with the
cu_seqlens offsets and runs a plain causal softmax(q·kᵀ/√d)·v in fp32 per sequence
(python loop over batch, full materialized scores). Correct, but slow. It is the
CORRECTNESS oracle.

        ## Iteration loop

        - Edit `/app/candidate.py`; PyTorch, Triton, CUDA headers, `nvcc`, and Ninja are available.
        - Run `python /app/evaluate.py` for a one-shape quick check.
        - Run `python /app/evaluate.py --full` for the complete public workload.
        - Repeat for the full two-hour Agent budget.
        - The FLA or FlashInfer baseline named in `reference_model.baseline()` is documentation and
          calibration metadata; those external packages are not installed in the task image.

        ## Workload

        - `[[64, 128, 96, 200], 32, 8, 128]`
- `[[256, 384, 128, 512], 32, 8, 128]`
- `[[512, 768, 256, 1024, 640], 32, 8, 128]`
- `[[1024, 1536, 768, 2048, 512, 900], 40, 8, 128]`

        Final verification uses hidden random tensor values for every public shape. It checks the
        selected forward output only; backward gradients are not part of this source suite's verifier.

        ## Scoring

        The score is the geometric mean of `naive_latency / candidate_latency` over all shapes.
        Correctness failure on any shape gives zero reward. The naive anchor is `1.0`; the calibrated
        `flashinfer` anchor is `5.7`. Reward is:

        ```text
        clip(0.5 * log(score) / log(5.7), 0, 1)
        ```

        ## Rules

        - Preserve `ModelNew` and support every listed shape.
        - Do not modify `reference_model.py`, `scorer.py`, or `evaluate.py`.
        - Do not inspect hidden verifier state or hard-code outputs.
        - Do not use the network or additional GPUs.
