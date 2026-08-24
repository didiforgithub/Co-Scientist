"""K3 task: Based (bigdim) — 2nd-order Taylor linear attention, large V head dim.

The fla fused_chunk_based kernel asserts feature dim K <= 16, so the query/key
Taylor-feature dim stays 16; this variant enlarges the VALUE head dim to 256.

Based (Zoology / "Simple linear attention language models", arXiv:2402.18668)
approximates softmax attention with the 2nd-order Taylor expansion of exp:
    exp(qᵀk) ≈ 1 + qᵀk + (qᵀk)²/2
This is a linear-attention operator: the feature map phi(x) turns attention
into a running sum of outer products of feature vectors, so it can be evaluated
with a causal recurrent state instead of the O(T²) attention matrix. There is no
native torch op for the chunkwise Taylor-feature scan — it must be written by
hand, which is what makes it a genuine kernel-research task.

Math (per head, verified against fla.ops.based.naive.naive_parallel_based):
    q <- q * scale,  scale = 1/sqrt(K)
    A_{ts} = 1 + qᵀ_t k_s + 0.5 (qᵀ_t k_s)²     for s <= t   (causal, inclusive)
    o_t = (sum_{s<=t} A_{ts} v_s) / (sum_{s<=t} A_{ts} + eps)   # use_norm=True
The numerator/denominator are accumulated with running Taylor moments:
    S0 = sum v_s               (zero-order, coeff 1)
    S1 = sum k_s ⊗ v_s         (first-order)
    S2 = sum (k_s⊗k_s) ⊗ v_s   (second-order, coeff 1/2)
and the matching key moments Z1, Z2 for the denominator.

This Model.forward IS the naive O(T) recurrent reference: correct and
differentiable, but slow (a Python loop over T). It is the CORRECTNESS oracle.

STRONG BASELINE to beat (not naive torch — that is a fake bar):
    fla.ops.based.fused_chunk_based  (SOTA Triton, chunkwise Taylor scan).
NOTE: the fused kernel asserts feature dim K <= 16.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Naive O(T) recurrent Based linear attention. Correct & differentiable, slow.

    Inputs (batch-first, matching fla convention with head_first=False):
      q, k : (B, T, H, K)   query/key,  K <= 16 (Taylor feature dim)
      v    : (B, T, H, V)   value
    Output:
      o    : (B, T, H, V)
    """

    def __init__(self, use_norm: bool = True):
        super().__init__()
        self.use_norm = use_norm

    def forward(self, q, k, v):
        B, T, H, K = q.shape
        V = v.shape[-1]
        idt = q.dtype
        q = q.transpose(1, 2).float()          # (B, H, T, K)
        k = k.transpose(1, 2).float()
        v = v.transpose(1, 2).float()          # (B, H, T, V)
        scale = 1.0 / (K ** 0.5)
        q = q * scale

        S0 = torch.zeros(B, H, V, dtype=torch.float32, device=q.device)
        S1 = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
        S2 = torch.zeros(B, H, K, K, V, dtype=torch.float32, device=q.device)
        Z1 = torch.zeros(B, H, K, dtype=torch.float32, device=q.device)
        Z2 = torch.zeros(B, H, K, K, dtype=torch.float32, device=q.device)
        out = torch.zeros(B, H, T, V, dtype=torch.float32, device=q.device)
        for t in range(T):
            q_t = q[:, :, t]                   # (B, H, K)
            k_t = k[:, :, t]                   # (B, H, K)
            v_t = v[:, :, t]                   # (B, H, V)
            kk = k_t[:, :, :, None] * k_t[:, :, None, :]   # (B, H, K, K)
            # accumulate causal-inclusive moments
            S0 = S0 + v_t
            S1 = S1 + k_t[:, :, :, None] * v_t[:, :, None, :]
            S2 = S2 + kk[:, :, :, :, None] * v_t[:, :, None, None, :]
            Z1 = Z1 + k_t
            Z2 = Z2 + kk
            qq = q_t[:, :, :, None] * q_t[:, :, None, :]   # (B, H, K, K)
            # numerator
            o_t = S0.clone()
            o_t = o_t + (q_t[:, :, :, None] * S1).sum(dim=-2)
            o_t = o_t + 0.5 * (qq[:, :, :, :, None] * S2).sum(dim=(-3, -2))
            if self.use_norm:
                z_t = (t + 1.0) + (q_t * Z1).sum(-1) + 0.5 * (qq * Z2).sum(dim=(-2, -1))
                o_t = o_t / (z_t[:, :, None] + 1e-6)
            out[:, :, t] = o_t
        return out.transpose(1, 2).contiguous().to(idt)   # (B, T, H, V)


# --- workload distribution --- BIG-HEAD-DIM variant (V=256; K capped at 16).
WORKLOAD = [
    (1, 128, 4, 16, 256),
    (2, 256, 4, 16, 256),
    (1, 512, 8, 16, 256),
    (2, 256, 8, 16, 256),
]

_DEF = (1, 128, 4, 16, 256)


def get_inputs():
    torch.manual_seed(0)
    B, T, H, K, V = _DEF
    q = torch.randn(B, T, H, K, dtype=torch.float32) * 0.5
    k = torch.randn(B, T, H, K, dtype=torch.float32) * 0.5
    v = torch.randn(B, T, H, V, dtype=torch.float32) * 0.5
    return [q, k, v]


def get_init_inputs():
    return [True]  # use_norm


# --- generic-framework contract (make_inputs / baseline) ---
def make_inputs(shape, device, seed=0):
    B, T, H, K, V = shape
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, device=device) * 0.5
    k = torch.randn(B, T, H, K, device=device) * 0.5
    v = torch.randn(B, T, H, V, device=device) * 0.5
    return [q, k, v]


OUTPUT_INDEX = 0  # forward returns the output tensor directly


def baseline(q, k, v, init_args=None):
    """Strong baseline: fla fused_chunk_based (SOTA Triton, chunkwise Taylor)."""
    from fla.ops.based import fused_chunk_based
    K = q.shape[-1]
    use_norm = (init_args[0] if init_args else True)
    o = fused_chunk_based(q, k, v, scale=1.0 / (K ** 0.5),
                          use_norm=use_norm, head_first=False)
    return o
