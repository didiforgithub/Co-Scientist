        # K3_gated_oja_rule_bf16

        Optimize `/app/candidate.py` for a single NVIDIA H100. It must define
        `class ModelNew` with the same initialization and forward contract as
        `/app/reference_model.py`.

        ## Background

        K3 task: Gated Oja Rule — Oja's-rule linear attention with a value-channel gate.

The gated Oja rule is a delta-rule variant where the correction is applied on the
*key* side (Oja's Hebbian learning rule keeps the state's rows bounded) and the
state decay `gv` acts per *value* channel. `HV` value heads may share `H`
query/key heads (GVA). There is NO native torch op — the chunkwise WY scan must
be hand-written.

Math (per value-head, verified against fla.ops.gated_oja_rule fused_recurrent_oja kernel):
    q, k <- L2norm(q), L2norm(k)          # use_q_l2norm / use_k_l2norm
    q <- q * scale,  scale = 1/sqrt(K)
    for t in 0..T:
        S     <- S * exp(gv_t)[None,:]    # per-V-channel log decay (value side)
        k_new <- beta_t * (k_t - S v_t)   # Oja correction, read S@v (contract V)
        S     <- S + k_new ⊗ v_t          # rank-1 write on the KEY side
        o_t   <- qᵀ_t S                    # readout
    gv_t: log gate of shape [V];  S: state [K, V] per (batch, value-head)

This Model.forward IS the naive O(T) recurrent oracle: correct, differentiable, slow.

        ## Iteration loop

        - Edit `/app/candidate.py`; PyTorch, Triton, CUDA headers, `nvcc`, and Ninja are available.
        - Run `python /app/evaluate.py` for a one-shape quick check.
        - Run `python /app/evaluate.py --full` for the complete public workload.
        - Repeat for the full two-hour Agent budget.
        - The FLA or FlashInfer baseline named in `reference_model.baseline()` is documentation and
          calibration metadata; those external packages are not installed in the task image.

        ## Workload

        - `[1, 128, 4, 4, 64, 64]`
- `[2, 256, 8, 8, 128, 128]`
- `[1, 512, 8, 8, 128, 128]`
- `[2, 256, 4, 4, 64, 64]`

        Final verification uses hidden random tensor values for every public shape. It checks the
        selected forward output only; backward gradients are not part of this source suite's verifier.

        ## Scoring

        The score is the geometric mean of `naive_latency / candidate_latency` over all shapes.
        Correctness failure on any shape gives zero reward. The naive anchor is `1.0`; the calibrated
        `fla.gated_oja_rule:chunk_gated_oja_rule` anchor is `38.4`. Reward is:

        ```text
        clip(0.5 * log(score) / log(38.4), 0, 1)
        ```

        ## Rules

        - Preserve `ModelNew` and support every listed shape.
        - Do not modify `reference_model.py`, `scorer.py`, or `evaluate.py`.
        - Do not inspect hidden verifier state or hard-code outputs.
        - Do not use the network or additional GPUs.
