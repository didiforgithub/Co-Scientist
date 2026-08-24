        # K3_mesa_net_longseq

        Optimize `/app/candidate.py` for a single NVIDIA H100. It must define
        `class ModelNew` with the same initialization and forward contract as
        `/app/reference_model.py`.

        ## Background

        K3 task: MesaNet — test-time regression (mesa-optimization) linear attention.

MesaNet replaces the delta-rule's single gradient step with an *exact* least-
squares solve at every position: the readout solves (H_kk + diag(lamb)) x = q,
then reads o = x·H_kv, where H_kk = sum of decayed β·k⊗k and H_kv = sum of
decayed β·k⊗v. The fla Triton kernel approximates the solve with `max_CG_iteration`
conjugate-gradient steps. There is NO native torch op — the chunkwise scan plus
in-kernel CG solve is a genuinely hard hand-written kernel.

Math (per head, verified against fla.ops.mesa_net.naive.naive_mesa_net_exact):
    for t in 0..T:
        H_kk <- H_kk*exp(g_t) + (beta_t*k_t) ⊗ k_t     # [K,K]
        H_kv <- H_kv*exp(g_t) + (beta_t*k_t) ⊗ v_t     # [K,V]
    x_t  = solve(H_kk(t) + diag(lamb), q_t)             # ridge-regularized solve
    o_t  = x_t · H_kv(t)                                # readout
    g_t: head-wise log decay;  beta_t: head-wise gate;  lamb: [H,K] ridge (>=0.25)

This Model.forward IS the exact O(T) oracle (linalg.solve), correct & slow.
Note: the baseline uses 30 CG iterations, so agreement is at CG-convergence tol.

        ## Iteration loop

        - Edit `/app/candidate.py`; PyTorch, Triton, CUDA headers, `nvcc`, and Ninja are available.
        - Run `python /app/evaluate.py` for a one-shape quick check.
        - Run `python /app/evaluate.py --full` for the complete public workload.
        - Repeat for the full two-hour Agent budget.
        - The FLA or FlashInfer baseline named in `reference_model.baseline()` is documentation and
          calibration metadata; those external packages are not installed in the task image.

        ## Workload

        - `[1, 1024, 4, 64, 64]`
- `[2, 2048, 4, 64, 64]`
- `[1, 4096, 4, 64, 64]`

        Final verification uses hidden random tensor values for every public shape. It checks the
        selected forward output only; backward gradients are not part of this source suite's verifier.

        ## Scoring

        The score is the geometric mean of `naive_latency / candidate_latency` over all shapes.
        Correctness failure on any shape gives zero reward. The naive anchor is `1.0`; the calibrated
        `fla.mesa_net:chunk_mesa_net` anchor is `743.3`. Reward is:

        ```text
        clip(0.5 * log(score) / log(743.3), 0, 1)
        ```

        ## Rules

        - Preserve `ModelNew` and support every listed shape.
        - Do not modify `reference_model.py`, `scorer.py`, or `evaluate.py`.
        - Do not inspect hidden verifier state or hard-code outputs.
        - Do not use the network or additional GPUs.
