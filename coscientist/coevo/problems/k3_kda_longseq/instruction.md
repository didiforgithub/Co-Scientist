        # K3_kda_longseq

        Optimize `/app/candidate.py` for a single NVIDIA H100. It must define
        `class ModelNew` with the same initialization and forward contract as
        `/app/reference_model.py`.

        ## Background

        K3 task: KDA (Kimi Delta Attention) — delta rule with a per-channel forget gate.

KDA is the Kimi-linear delta-rule variant. Like gated DeltaNet it does a rank-1
delta-rule state update, but the state decay is a *per-K-channel* log gate `g`
(as in GLA) instead of a single head-wise scalar (as in GDN). It also uses GVA
(grouped value attention): `HV` value heads share `H` query/key heads. There is
NO native torch op — the chunkwise WY-representation scan is hand-written.

Math (per value-head, verified against fla.ops.kda.naive.naive_recurrent_kda + l2norm):
    q, k <- L2norm(q), L2norm(k)          # use_qk_l2norm_in_kernel=True
    q <- q * scale,  scale = 1/sqrt(K)
    for t in 0..T:
        S     <- S * exp(g_t)[:,None]     # per-K-channel log decay
        delta <- (v_t - Sᵀ k_t) * beta_t  # delta-rule correction
        S     <- S + k_t ⊗ delta          # rank-1 write
        o_t   <- qᵀ_t S                    # readout
    g_t: log-space per-channel gate of shape [K];  S: state [K, V] per (b, v-head)

This Model.forward IS the naive O(T) recurrent oracle: correct, differentiable, slow.

        ## Iteration loop

        - Edit `/app/candidate.py`; PyTorch, Triton, CUDA headers, `nvcc`, and Ninja are available.
        - Run `python /app/evaluate.py` for a one-shape quick check.
        - Run `python /app/evaluate.py --full` for the complete public workload.
        - Repeat for the full two-hour Agent budget.
        - The FLA or FlashInfer baseline named in `reference_model.baseline()` is documentation and
          calibration metadata; those external packages are not installed in the task image.

        ## Workload

        - `[1, 1024, 8, 16, 128, 128]`
- `[1, 2048, 8, 16, 128, 128]`

        Final verification uses hidden random tensor values for every public shape. It checks the
        selected forward output only; backward gradients are not part of this source suite's verifier.

        ## Scoring

        The score is the geometric mean of `naive_latency / candidate_latency` over all shapes.
        Correctness failure on any shape gives zero reward. The naive anchor is `1.0`; the calibrated
        `fla.kda:chunk_kda` anchor is `137.1`. Reward is:

        ```text
        clip(0.5 * log(score) / log(137.1), 0, 1)
        ```

        ## Rules

        - Preserve `ModelNew` and support every listed shape.
        - Do not modify `reference_model.py`, `scorer.py`, or `evaluate.py`.
        - Do not inspect hidden verifier state or hard-code outputs.
        - Do not use the network or additional GPUs.
