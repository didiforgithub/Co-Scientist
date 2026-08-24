"""K3 task: DPLR — Diagonal-Plus-Low-Rank generalized delta rule.

DPLR (fla.ops.generalized_delta_rule.dplr) generalizes the delta rule: instead
of a scalar/diagonal state decay, the recurrent state is transformed each step by
a diagonal-plus-rank-1 matrix  Diag(exp(gk)) + a bᵀ  before the k⊗v write:
    S_t = Diag(exp(gk_t)) S_{t-1} + a_t (b_tᵀ S_{t-1}) + k_t ⊗ v_t
        = exp(gk_t) ⊙ S_{t-1} + a_t ⊗ (S_{t-1}ᵀ a_t? )  ... (see exact form below)
    o_t = q_tᵀ S_t
This is the RWKV-7 / generalized-delta family. There is NO native torch op for
the chunkwise DPLR scan — it must be written by hand, which is what makes it a
genuine kernel-research task.

Exact per-step recurrence (verified against fla.ops...dplr.naive.dplr_recurrence
and the fused_recurrent kernel):
    q <- q * scale,  scale = 1/sqrt(K)
    for t in 0..T:
        kv = k_t ⊗ v_t + b_t ⊗ (a_tᵀ S_{t-1})      # low-rank correction via a,b
        S  = exp(gk_t)[:, None] ⊙ S_{t-1} + kv     # diagonal decay + write
        o_t = q_tᵀ S
    S: recurrent state [K, V] per (batch, head)

This Model.forward IS the naive O(T) recurrent reference: correct and
differentiable, but slow (a Python loop over T). It is the CORRECTNESS oracle.

STRONG BASELINE to beat (not naive torch — that is a fake bar):
    fla.ops.generalized_delta_rule.dplr.chunk_dplr_delta_rule (SOTA Triton).
NOTE: the chunk DPLR kernel does not support fp32; inputs are cast to bfloat16
and the tolerance is 2e-2.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Naive O(T) recurrent DPLR generalized delta rule. Correct, slow.

    Inputs (batch-first, matching fla convention):
      q, k : (B, T, H, K)   query/key
      v    : (B, T, H, V)   value
      a    : (B, T, H, K)   low-rank vector a (aka alpha)
      b    : (B, T, H, K)   low-rank vector b (aka beta)
      gk   : (B, T, H, K)   log-space per-channel diagonal decay (<= 0)
    Output:
      o    : (B, T, H, V)
    """

    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, a, b, gk):
        B, T, H, K = q.shape
        V = v.shape[-1]
        idt = q.dtype
        q = q.transpose(1, 2).float()          # (B, H, T, K)
        k = k.transpose(1, 2).float()
        v = v.transpose(1, 2).float()          # (B, H, T, V)
        a = a.transpose(1, 2).float()
        b = b.transpose(1, 2).float()
        gk = gk.transpose(1, 2).float()
        scale = 1.0 / (K ** 0.5)
        q = q * scale

        S = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
        out = torch.zeros(B, H, T, V, dtype=torch.float32, device=q.device)
        for t in range(T):
            q_t = q[:, :, t]                   # (B, H, K)
            k_t = k[:, :, t]                   # (B, H, K)
            v_t = v[:, :, t]                   # (B, H, V)
            a_t = a[:, :, t]                   # (B, H, K)
            b_t = b[:, :, t]                   # (B, H, K)
            g_t = gk[:, :, t].exp()            # (B, H, K)
            # low-rank term: b_t ⊗ (a_tᵀ S) ; (a_tᵀ S) is (B, H, V)
            aS = (S * a_t[:, :, :, None]).sum(dim=-2)           # (B, H, V)
            kv = k_t[:, :, :, None] * v_t[:, :, None, :] \
                + b_t[:, :, :, None] * aS[:, :, None, :]        # (B, H, K, V)
            S = S * g_t[:, :, :, None] + kv
            out[:, :, t] = (S * q_t[:, :, :, None]).sum(dim=-2)  # qᵀS
        return out.transpose(1, 2).contiguous().to(idt)   # (B, T, H, V)


# --- workload distribution --- (B, T, H, K, V). chunk=16 (DPLR kernel default).
WORKLOAD = [
    (1, 128, 4, 64, 64),
    (2, 256, 4, 128, 128),
    (1, 512, 8, 128, 128),
    (2, 512, 8, 128, 128),
]

_DEF = (1, 128, 4, 64, 64)


def _make(B, T, H, K, V, device, seed):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, device=device) * 0.5
    k = torch.randn(B, T, H, K, device=device) * 0.5
    v = torch.randn(B, T, H, V, device=device) * 0.5
    # a, b low-rank vectors: keep small so the low-rank correction is well-conditioned
    a = torch.randn(B, T, H, K, device=device) * 0.1
    b = torch.randn(B, T, H, K, device=device) * 0.1
    # gk log-space per-channel decay: small negative
    gk = -torch.rand(B, T, H, K, device=device) * 0.1 - 0.01
    return [q, k, v, a, b, gk]


def get_inputs():
    B, T, H, K, V = _DEF
    return _make(B, T, H, K, V, "cpu", 0)


def get_init_inputs():
    return []


# --- generic-framework contract (make_inputs / baseline) ---
def make_inputs(shape, device, seed=0):
    B, T, H, K, V = shape
    return _make(B, T, H, K, V, device, seed)


OUTPUT_INDEX = 0  # forward returns the output tensor directly


def baseline(q, k, v, a, b, gk, init_args=None):
    """Strong baseline: fla chunk_dplr_delta_rule (SOTA Triton, chunkwise).

    NOTE: the DPLR chunk kernel does not support fp32 — cast to bf16 (tol 2e-2).
    """
    from fla.ops.generalized_delta_rule.dplr import chunk_dplr_delta_rule
    K = q.shape[-1]
    qb, kb, vb, ab, bb, gkb = (x.to(torch.bfloat16) for x in (q, k, v, a, b, gk))
    o, _ = chunk_dplr_delta_rule(qb, kb, vb, ab, bb, gkb,
                                 scale=1.0 / (K ** 0.5),
                                 output_final_state=False)
    return o.to(q.dtype)
