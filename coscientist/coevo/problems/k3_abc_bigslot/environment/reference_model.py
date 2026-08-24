"""K3 task: ABC (Attention with Bounded-memory Control) — slot-memory linear attn.

ABC routes information through a small set of `M` memory slots via two stacked
linear-attention recurrences. Unlike GSA, ABC has NO separately supplied forget
gate: the slot gate `g` and the normalized slot weights are *derived from the raw
slot scores `s`* via a log-cumsum-exp / softmax-in-time trick. There is NO native
torch op — the two chunkwise scans plus the in-kernel gate derivation must be
written by hand.

Math (per head, verified against fla.ops.abc.naive.naive_recurrent_abc, g=None path):
    scale = 1/sqrt(K)
    # derive slot gate & normalized slot weights from raw scores s (over time):
    z   = logcumsumexp(s, dim=T)                         # running log-normalizer
    g   = cat(z[:1], z[:-1]) - z    (shifted)            # log slot forget gate
    s'  = exp(s - z)                                     # time-normalized slots
    # stage 1 — write real keys into slot memory hk in R^{K x M}, read with q:
    for t: hk <- hk*exp(g_t)[None,:] + k_t ⊗ s'_t ;  ok_t = q_t·hk    # -> R^M
    qv = softmax(ok, dim=slots)
    # stage 2 — write real values into slot memory hv in R^{M x V}, read with qv:
    for t: hv <- hv*exp(g_t)[:,None] + s'_t ⊗ v_t ;  o_t = qv_t·hv    # -> R^V
    hk: R^{K x M}, hv: R^{M x V} recurrent states per (batch, head)

This Model.forward IS the naive O(T) recurrent oracle: correct, differentiable, slow.

STRONG BASELINE to beat: fla.ops.abc.chunk_abc  (SOTA Triton, chunkwise).
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Naive O(T) two-stage ABC slot-memory recurrence. Correct & differentiable, slow.

    Inputs (batch-first, no GQA so HQ == H):
      q, k : (B, T, H, K)   query / key
      v    : (B, T, H, V)   value
      s    : (B, T, H, M)   raw slot scores (gate + slot weights derived from this)
    Output:
      o    : (B, T, H, V)
    """

    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, s):
        dtype = q.dtype
        q, k, v, s = (x.transpose(1, 2).contiguous().float() for x in (q, k, v, s))
        B, H, T, K = q.shape
        V = v.shape[-1]
        M = s.shape[-1]
        scale = K ** -0.5

        # derive log-space slot gate g and time-normalized slot weights s' from raw s
        z = s.logcumsumexp(2)                               # (B,H,T,M)
        g = torch.cat((z[:, :, :1], z[:, :, :-1]), 2) - z   # (B,H,T,M) shifted
        s = torch.exp(s - z)                                # (B,H,T,M) normalized slots

        hk = torch.zeros(B, H, K, M, dtype=torch.float32, device=q.device)
        ok = torch.zeros_like(s)
        for i in range(T):
            q_i = q[:, :, i] * scale       # (B,H,K)
            k_i = k[:, :, i]               # (B,H,K)
            v_i = s[:, :, i]               # (B,H,M) normalized slot as value
            g_i = g[:, :, i].exp()         # (B,H,M)
            hk = hk * g_i[..., None, :] + k_i[..., None] * v_i[..., None, :]
            ok[:, :, i] = (q_i[..., None] * hk).sum(-2)   # (B,H,M)

        qv = ok.softmax(-1)
        hv = torch.zeros(B, H, M, V, dtype=torch.float32, device=q.device)
        ov = torch.zeros_like(v)
        for i in range(T):
            q_i = qv[:, :, i]              # (B,H,M)
            k_i = s[:, :, i]              # (B,H,M) normalized slot as key
            v_i = v[:, :, i]              # (B,H,V)
            g_i = g[:, :, i].exp()        # (B,H,M)
            hv = hv * g_i[..., :, None] + k_i[..., None] * v_i[..., None, :]
            ov[:, :, i] = (q_i[..., None] * hv).sum(-2)   # (B,H,V)

        return ov.transpose(1, 2).contiguous().to(dtype)


# --- workload distribution ---
# (B, T, H, K, V, M). K=V=head dim, M=slot count (bottleneck).
WORKLOAD = [
    (1, 128, 4, 64, 64, 128),
    (2, 256, 4, 128, 128, 128),
    (1, 512, 4, 128, 128, 128),
]

_DEF = (1, 128, 4, 64, 64, 128)


def _rand(shape, device, seed):
    torch.manual_seed(seed)
    B, T, H, K, V, M = shape
    q = torch.randn(B, T, H, K, device=device) * 0.5
    k = torch.randn(B, T, H, K, device=device) * 0.5
    v = torch.randn(B, T, H, V, device=device) * 0.5
    s = torch.randn(B, T, H, M, device=device) * 0.5   # raw slot scores
    return [q, k, v, s]


def get_inputs():
    return _rand(_DEF, "cpu", 0)


def get_init_inputs():
    return []


def make_inputs(shape, device, seed=0):
    return _rand(shape, device, seed)


OUTPUT_INDEX = 0


def baseline(q, k, v, s, init_args=None):
    """Strong baseline: fla chunk_abc (SOTA Triton). Gate derived internally from s."""
    from fla.ops.abc import chunk_abc
    o, _ = chunk_abc(q, k, v, s, output_final_state=False, head_first=False)
    return o
