"""K3 task: KDA (Kimi Delta Attention) — delta rule with a per-channel forget gate.

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

STRONG BASELINE to beat: fla.ops.kda.chunk_kda  (SOTA Triton, chunkwise).
"""
import torch
import torch.nn as nn


def _l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


class Model(nn.Module):
    """Naive O(T) recurrent KDA (GVA delta rule, per-channel gate). Slow oracle.

    Inputs (batch-first, matching fla.ops.kda convention):
      q, k : (B, T, H,  K)   query/key,  H  key/query heads
      v    : (B, T, HV, V)   value,      HV value heads (HV = H * groups, GVA)
      g    : (B, T, HV, K)   log-space per-channel forget gate (<= 0)
      beta : (B, T, HV)      delta update gate (already in [0,1))
    Output:
      o    : (B, T, HV, V)
    """

    def __init__(self, use_qk_l2norm: bool = True):
        super().__init__()
        self.use_qk_l2norm = use_qk_l2norm

    def forward(self, q, k, v, g, beta):
        B, T, H, K = q.shape
        HV, V = v.shape[2], v.shape[3]
        G = HV // H
        idt = v.dtype
        if self.use_qk_l2norm:
            q = _l2norm(q, dim=-1)
            k = _l2norm(k, dim=-1)
        q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
        scale = 1.0 / (K ** 0.5)
        # expand q/k to value-head count (GVA): [B,T,H,K] -> [B,T,HV,K]
        q = q.repeat_interleave(G, dim=2) * scale
        k = k.repeat_interleave(G, dim=2)

        S = torch.zeros(B, HV, K, V, dtype=torch.float32, device=q.device)
        o = torch.zeros(B, T, HV, V, dtype=torch.float32, device=q.device)
        for i in range(T):
            q_i, k_i, v_i = q[:, i], k[:, i], v[:, i]      # (B,HV,K),(B,HV,K),(B,HV,V)
            g_i, b_i = g[:, i], beta[:, i]                 # (B,HV,K),(B,HV)
            S = S * g_i[..., None].exp()                   # per-channel decay
            kv_mem = (k_i[..., None] * S).sum(-2)          # Sᵀk : (B,HV,V)
            delta = (v_i - kv_mem) * b_i[..., None]        # (B,HV,V)
            S = S + k_i[..., None] * delta[..., None, :]   # rank-1 write
            o[:, i] = (q_i[..., None] * S).sum(-2)         # qᵀS
        return o.to(idt)


# --- workload distribution --- BF16 variant. (B, T, H, HV, K, V).
# Inputs are bf16; tensor-core utilization in the GVA chunkwise delta matmuls is
# the optimization target. Tolerance is relaxed to 2e-2.
WORKLOAD = [
    (1, 128, 4, 8, 64, 64),
    (2, 256, 8, 16, 128, 128),
    (1, 512, 8, 16, 128, 128),
    (2, 256, 4, 8, 64, 64),
]
_DEF_B = (1, 128, 4, 8, 64, 64)

_DEF = _DEF_B


def _rand(shape, device, seed):
    torch.manual_seed(seed)
    B, T, H, HV, K, V = shape
    q = torch.randn(B, T, H, K, device=device) * 0.5
    k = torch.randn(B, T, H, K, device=device) * 0.5
    v = torch.randn(B, T, HV, V, device=device) * 0.5
    # log-space per-channel decay, small negative (state slowly decays)
    g = -torch.rand(B, T, HV, K, device=device) * 0.1 - 0.01
    beta = torch.rand(B, T, HV, device=device)   # update gate in [0,1)
    return [q, k, v, g, beta]


def get_inputs():
    return _rand(_DEF, "cpu", 0)


def get_init_inputs():
    return [True]  # use_qk_l2norm


def make_inputs(shape, device, seed=0):
    return [x.to(torch.bfloat16) for x in _rand(shape, device, seed)]


OUTPUT_INDEX = 0


def baseline(q, k, v, g, beta, init_args=None):
    """Strong baseline: fla chunk_kda (SOTA Triton, chunkwise)."""
    from fla.ops.kda import chunk_kda
    K = q.shape[-1]
    l2 = (init_args[0] if init_args else True)
    o, _ = chunk_kda(q, k, v, g, beta, scale=1.0 / (K ** 0.5),
                     output_final_state=False,
                     use_qk_l2norm_in_kernel=l2,
                     use_gate_in_kernel=False)
    return o
