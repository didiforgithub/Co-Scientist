"""K3 task: Gated Slot Attention (GSA) — chunkwise linear attention with a slot bottleneck.

GSA routes information through a small set of `M` memory slots via two stacked
gated linear-attention recurrences (an ABC-style bottleneck). There is NO native
torch op — the two chunkwise scans over the slot dimension must be written by hand.

Math (per head, verified against fla.ops.gsa.naive.naive_recurrent_gsa):
    scale = 1/sqrt(K),  q <- q * scale
    # stage 1 — write real keys into slot memory hk in R^{K x M}, read with q:
    for t: hk <- hk * exp(g_t)[None,:] + k_t ⊗ s_t ;  ok_t = (q_t · hk)   # -> R^M
    qv = softmax(ok, dim=slots)                                            # slot attention
    # stage 2 — write real values into slot memory hv in R^{M x V}, read with qv:
    for t: hv <- hv * exp(g_t)[:,None] + s_t ⊗ v_t ;  o_t = (qv_t · hv)    # -> R^V
    hk: R^{K x M}, hv: R^{M x V} recurrent states per (batch, head)

`g` is the log-space forget gate on the M slots. `s` is the slot representation
(acts as value in stage 1 and as key in stage 2). This Model.forward IS the naive
O(T) recurrent oracle: correct, differentiable, slow.

STRONG BASELINE to beat: fla.ops.gsa.chunk_gsa (SOTA Triton, chunkwise).
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Naive O(T) two-stage gated slot-attention recurrence.

    Inputs (batch-first, HQ == H so no GQA):
      q, k : (B, T, H, K)   query / key
      v    : (B, T, H, V)   value
      s    : (B, T, H, M)   slot representation (value in stage 1, key in stage 2)
      g    : (B, T, H, M)   log-space forget gate on the M slots
    Output:
      o    : (B, T, H, V)
    """

    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, s, g):
        dtype = q.dtype
        q, k, v, s, g = (x.transpose(1, 2).contiguous().float() for x in (q, k, v, s, g))
        B, H, T, K = q.shape
        V = v.shape[-1]
        M = s.shape[-1]
        scale = K ** -0.5

        hk = torch.zeros(B, H, K, M, dtype=torch.float32, device=q.device)
        ok = torch.zeros_like(s)
        for i in range(T):
            q_i = q[:, :, i] * scale       # (B,H,K)
            k_i = k[:, :, i]               # (B,H,K)
            v_i = s[:, :, i]               # (B,H,M) slot as value
            g_i = g[:, :, i].exp()         # (B,H,M)
            hk = hk * g_i[..., None, :] + k_i[..., None] * v_i[..., None, :]
            ok[:, :, i] = (q_i[..., None] * hk).sum(-2)   # (B,H,M)

        qv = ok.softmax(-1)
        hv = torch.zeros(B, H, M, V, dtype=torch.float32, device=q.device)
        ov = torch.zeros_like(v)
        for i in range(T):
            q_i = qv[:, :, i]              # (B,H,M)
            k_i = s[:, :, i]              # (B,H,M) slot as key
            v_i = v[:, :, i]              # (B,H,V)
            g_i = g[:, :, i].exp()        # (B,H,M)
            hv = hv * g_i[..., :, None] + k_i[..., None] * v_i[..., None, :]
            ov[:, :, i] = (q_i[..., None] * hv).sum(-2)   # (B,H,V)

        return ov.transpose(1, 2).contiguous().to(dtype)


# --- workload distribution --- BF16 variant.
# (B, T, H, K, V, M). K=V=head dim, M=slot count (bottleneck < head dim).
# NOTE: fla chunk_gsa is bf16-unstable at K=128/M=64 (produces NaN), so the bf16
# variant uses K=64/M=32 slot configs where the bf16 tensor-core path is stable.
WORKLOAD = [
    (1, 128, 4, 64, 64, 32),   # small smoke
    (2, 256, 4, 64, 64, 32),   # medium
    (1, 512, 8, 64, 64, 32),   # longer seq
    (2, 256, 8, 64, 64, 32),   # more heads
]

_DEF = (1, 128, 4, 64, 64, 32)


def _rand(shape, device, seed):
    torch.manual_seed(seed)
    B, T, H, K, V, M = shape
    q = torch.randn(B, T, H, K, device=device) * 0.5
    k = torch.randn(B, T, H, K, device=device) * 0.5
    v = torch.randn(B, T, H, V, device=device) * 0.5
    s = torch.randn(B, T, H, M, device=device) * 0.5
    # log-space forget gate on slots (logsigmoid -> negative)
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, M, device=device))
    return [q, k, v, s, g]


def get_inputs():
    return _rand(_DEF, "cpu", 0)


def get_init_inputs():
    return []


# BF16 variant: inputs are bf16, so tensor-core (mma) utilization in the two
# chunkwise slot-attention scans is the optimization target. Tolerance 2e-2.
def make_inputs(shape, device, seed=0):
    return [x.to(torch.bfloat16) for x in _rand(shape, device, seed)]


OUTPUT_INDEX = 0


def baseline(q, k, v, s, g, init_args=None):
    """Strong baseline: fla chunk_gsa (SOTA Triton)."""
    from fla.ops.gsa import chunk_gsa
    K = q.shape[-1]
    o, _ = chunk_gsa(q, k, v, s, g, scale=K ** -0.5, output_final_state=False)
    return o
