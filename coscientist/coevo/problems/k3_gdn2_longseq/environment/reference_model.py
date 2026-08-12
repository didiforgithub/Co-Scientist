"""K3 task: Gated DeltaNet 2 (GDN-2) — chunkwise delta rule with channel-wise gates.

GDN-2 generalizes the gated delta rule with TWO channel-wise gates: an erase gate
`b` on the key axis and a write gate `w` on the value axis (plus per-channel log
decay `g`). Collapsing b = w = scalar recovers KDA. There is NO native torch op —
the chunkwise WY-representation scan is a genuine hand-written kernel.

Math (per head, on matrix state S in R^{K x V}, verified against
fla.ops.gdn2.naive.naive_recurrent_gdn2):
    scale = 1/sqrt(K),  q <- q * scale
    for t:
        S      <- Diag(exp(g_t)) S              # per-channel decay on K axis
        erase  <- ((b_t * k_t) · S)             # gated read at key  -> R^V
        v_new  <- w_t * v_t - erase             # gated write minus erase
        S      <- S + k_t ⊗ v_new               # rank-1 update
        o_t    <- (q_t · S)                     # readout            -> R^V
    equivalently S_t = (I - k_t (b_t*k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t*v_t)^T

`g` = channel-wise log-decay [.,.,H,K]; `b` = erase gate [.,.,H,K]; `w` = write
gate [.,.,H,V]. This Model.forward IS the naive O(T) recurrent oracle.

STRONG BASELINE to beat: fla.ops.gdn2.chunk_gdn2 (SOTA Triton, chunkwise).
"""
import torch
import torch.nn as nn


def _l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


class Model(nn.Module):
    """Naive O(T) GDN-2 recurrence. Correct & differentiable, slow.

    Inputs (batch-first):
      q, k : (B, T, H, K)   query / key
      v    : (B, T, H, V)   value
      g    : (B, T, H, K)   channel-wise log-decay on the key axis
      b    : (B, T, H, K)   channel-wise erase gate (key axis)
      w    : (B, T, H, V)   channel-wise write gate (value axis)
    Output:
      o    : (B, T, H, V)
    """

    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, g, b, w):
        dtype = v.dtype
        scale = q.shape[-1] ** -0.5
        # L2-normalize q,k (matches use_qk_l2norm_in_kernel=True; keeps the
        # delta-rule state well conditioned, as in production GDN layers).
        q = _l2norm(q, dim=-1)
        k = _l2norm(k, dim=-1)
        q, k, v, g, b, w = (x.transpose(1, 2).contiguous().float() for x in (q, k, v, g, b, w))
        B, H, T, K = k.shape
        V = v.shape[-1]
        o = torch.zeros(B, H, T, V, dtype=torch.float32, device=v.device)
        h = torch.zeros(B, H, K, V, dtype=torch.float32, device=v.device)
        q = q * scale
        for t in range(T):
            b_q = q[:, :, t]                          # (B,H,K)
            b_k = k[:, :, t]
            b_v = v[:, :, t]                          # (B,H,V)
            b_g = g[:, :, t]                          # (B,H,K)
            b_b = b[:, :, t]                          # (B,H,K)
            b_w = w[:, :, t]                          # (B,H,V)
            h = h * b_g.exp().unsqueeze(-1)           # decay on K axis
            erase = ((b_b * b_k).unsqueeze(-1) * h).sum(-2)   # (B,H,V)
            b_v_new = b_w * b_v - erase                        # (B,H,V)
            h = h + b_k.unsqueeze(-1) * b_v_new.unsqueeze(-2)  # rank-1
            o[:, :, t] = (b_q.unsqueeze(-1) * h).sum(-2)       # (B,H,V)
        return o.transpose(1, 2).contiguous().to(dtype)


# --- workload distribution --- LONG-SEQUENCE variant. (B, T, H, K, V).
# Large T (B=1, H=8, K=V=128): inter-chunk recurrence dominates cost, so the
# chunk-to-chunk state scan is the optimization target. T capped at 2048 so the
# naive O(T) recurrent oracle stays tractable.
WORKLOAD = [
    (1, 1024, 8, 128, 128),
    (1, 2048, 8, 128, 128),
]
_DEF_L = (1, 1024, 8, 128, 128)

_DEF = _DEF_L


def _rand(shape, device, seed):
    torch.manual_seed(seed)
    B, T, H, K, V = shape
    q = torch.randn(B, T, H, K, device=device) * 0.5
    k = torch.randn(B, T, H, K, device=device) * 0.5
    v = torch.randn(B, T, H, V, device=device) * 0.5
    # channel-wise log-decay (logsigmoid -> negative)
    g = torch.nn.functional.logsigmoid(torch.rand(B, T, H, K, device=device))
    b = torch.rand(B, T, H, K, device=device)      # erase gate in [0,1)
    w = torch.rand(B, T, H, V, device=device)      # write gate in [0,1)
    return [q, k, v, g, b, w]


def get_inputs():
    return _rand(_DEF, "cpu", 0)


def get_init_inputs():
    return []


def make_inputs(shape, device, seed=0):
    return _rand(shape, device, seed)


OUTPUT_INDEX = 0


def baseline(q, k, v, g, b, w, init_args=None):
    """Strong baseline: fla chunk_gdn2 (SOTA Triton)."""
    from fla.ops.gdn2 import chunk_gdn2
    K = q.shape[-1]
    o, _ = chunk_gdn2(q, k, v, g, b, w, scale=K ** -0.5, output_final_state=False,
                      use_qk_l2norm_in_kernel=True)
    return o
