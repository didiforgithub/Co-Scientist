"""K3 task: COMBA — gated delta rule with an auxiliary key `p`.

COMBA is a delta-rule variant where the *erase/correction* read uses a separate
auxiliary key `p` (instead of reusing `k`), while the rank-1 write still uses `k`.
This decouples "what to forget" from "what to write". There is NO native torch op —
the chunkwise WY-representation scan is a genuine hand-written kernel.

Math (per head, on matrix state h in R^{K x V}, verified against
fla.ops.comba.naive.naive_recurrent_comba):
    scale = 1/sqrt(K),  q <- q * scale
    for t:
        h      <- exp(g_t) * h                  # scalar (head-wise) log decay
        v_new  <- (v_t - (p_t · h)) * beta_t    # delta correction read with p
        h      <- h + k_t ⊗ v_new               # rank-1 write with k
        o_t    <- (q_t · h)                      # readout
    h: R^{K x V} recurrent state per (batch, head)

`g` = head-wise log-decay [.,.,H]; `beta` = update gate [.,.,H]; `p` = auxiliary
key [.,.,H,K]. This Model.forward IS the naive O(T) recurrent oracle.

STRONG BASELINE to beat: fla.ops.comba.chunk_comba (SOTA Triton, chunkwise).
"""
import torch
import torch.nn as nn


def _l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


class Model(nn.Module):
    """Naive O(T) COMBA recurrence. Correct & differentiable, slow.

    Inputs (batch-first):
      q, k : (B, T, H, K)   query / key (k drives the rank-1 write)
      v    : (B, T, H, V)   value
      p    : (B, T, H, K)   auxiliary key (drives the delta correction read)
      g    : (B, T, H)      head-wise log-decay
      beta : (B, T, H)      update gate
    Output:
      o    : (B, T, H, V)
    """

    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, p, g, beta):
        # L2-normalize q,k,p (matches use_qk_l2norm_in_kernel=True; keeps the
        # delta-rule state well conditioned).
        q = _l2norm(q, dim=-1)
        k = _l2norm(k, dim=-1)
        p = _l2norm(p, dim=-1)
        q, k, v, p, beta, g = (x.transpose(1, 2).contiguous().float()
                               for x in (q, k, v, p, beta, g))
        B, H, T, K = k.shape
        V = v.shape[-1]
        scale = K ** -0.5
        o = torch.zeros(B, H, T, V, dtype=torch.float32, device=v.device)
        h = torch.zeros(B, H, K, V, dtype=torch.float32, device=v.device)
        q = q * scale
        for i in range(T):
            b_q = q[:, :, i]                 # (B,H,K)
            b_k = k[:, :, i]
            b_v = v[:, :, i].clone()         # (B,H,V)
            b_p = p[:, :, i]                 # (B,H,K)
            h = h * g[:, :, i].exp()[..., None, None]
            b_beta = beta[:, :, i]           # (B,H)
            b_v = b_v - (h * b_p[..., None]).sum(-2)   # delta read with p
            b_v = b_v * b_beta[..., None]
            h = h + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)   # write with k
            o[:, :, i] = torch.einsum('bhd,bhdm->bhm', b_q, h)
        return o.transpose(1, 2).contiguous()


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
    p = torch.randn(B, T, H, K, device=device) * 0.5
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, device=device))
    beta = torch.rand(B, T, H, device=device)      # update gate in [0,1)
    return [q, k, v, p, g, beta]


def get_inputs():
    return _rand(_DEF, "cpu", 0)


def get_init_inputs():
    return []


def make_inputs(shape, device, seed=0):
    return _rand(shape, device, seed)


OUTPUT_INDEX = 0


def baseline(q, k, v, p, g, beta, init_args=None):
    """Strong baseline: fla chunk_comba (SOTA Triton)."""
    from fla.ops.comba import chunk_comba
    K = q.shape[-1]
    o, _ = chunk_comba(q, k, v, p, g, beta, scale=K ** -0.5, output_final_state=False,
                       use_qk_l2norm_in_kernel=True)
    return o
