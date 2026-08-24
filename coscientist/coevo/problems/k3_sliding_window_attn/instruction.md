        # K3_sliding_window_attn

        Optimize `/app/candidate.py` for a single NVIDIA H100. It must define
        `class ModelNew` with the same initialization and forward contract as
        `/app/reference_model.py`.

        ## Background

        K3 task: Sliding-window causal self-attention (local-attention prefill).

REAL production hotspot: models like Mistral / Gemma / Qwen-long use *sliding
window* attention — each query token attends only to itself and the previous W
keys (a local causal band), not the whole history. This bounds the KV footprint
and the attention cost at long context. The naive dense implementation still
materializes the full S×S score matrix and then throws most of it away with a
banded mask, which is exactly what a good kernel avoids.

This Model.forward IS the naive reference: plain softmax(q·kᵀ/√d)·v with an
explicit banded (causal AND within-window) mask, in fp32. Correct, but slow and
memory-heavy (full S×S scores). It is the CORRECTNESS oracle.

        ## Iteration loop

        - Edit `/app/candidate.py`; PyTorch, Triton, CUDA headers, `nvcc`, and Ninja are available.
        - Run `python /app/evaluate.py` for a one-shape quick check.
        - Run `python /app/evaluate.py --full` for the complete public workload.
        - Repeat for the full two-hour Agent budget.
        - The FLA or FlashInfer baseline named in `reference_model.baseline()` is documentation and
          calibration metadata; those external packages are not installed in the task image.

        ## Workload

        - `[512, 32, 8, 128]`
- `[1024, 32, 8, 128]`
- `[2048, 32, 8, 128]`
- `[4096, 40, 8, 128]`

        Final verification uses hidden random tensor values for every public shape. It checks the
        selected forward output only; backward gradients are not part of this source suite's verifier.

        ## Scoring

        The score is the geometric mean of `naive_latency / candidate_latency` over all shapes.
        Correctness failure on any shape gives zero reward. The naive anchor is `1.0`; the calibrated
        `flashinfer` anchor is `13.4`. Reward is:

        ```text
        clip(0.5 * log(score) / log(13.4), 0, 1)
        ```

        ## Rules

        - Preserve `ModelNew` and support every listed shape.
        - Do not modify `reference_model.py`, `scorer.py`, or `evaluate.py`.
        - Do not inspect hidden verifier state or hard-code outputs.
        - Do not use the network or additional GPUs.
