"""K3 task: Gated Oja Rule — Oja's-rule linear attention with a value-channel gate.

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

STRONG BASELINE to beat: fla.ops.gated_oja_rule.chunk_gated_oja_rule (SOTA Triton).
"""
import torch
import torch.nn as nn


def _l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


class Model(nn.Module):
    """Naive O(T) recurrent gated Oja rule. Correct & differentiable, slow.

    Inputs (batch-first, matching fla.ops.gated_oja_rule convention):
      q, k : (B, T, H,  K)   query/key,  H  key/query heads
      v    : (B, T, HV, V)   value,      HV value heads (GVA)
      gv   : (B, T, HV, V)   log-space per-V-channel forget gate (<= 0)
      beta : (B, T, HV)      head-wise Oja update gate
    Output:
      o    : (B, T, HV, V)
    """

    def __init__(self, use_q_l2norm: bool = True, use_k_l2norm: bool = True):
        super().__init__()
        self.use_q_l2norm = use_q_l2norm
        self.use_k_l2norm = use_k_l2norm

    def forward(self, q, k, v, gv, beta):
        B, T, H, K = q.shape
        HV, V = v.shape[2], v.shape[3]
        G = HV // H
        idt = v.dtype
        if self.use_q_l2norm:
            q = _l2norm(q, dim=-1)
        if self.use_k_l2norm:
            k = _l2norm(k, dim=-1)
        q, k, v, gv, beta = (x.float() for x in (q, k, v, gv, beta))
        scale = 1.0 / (K ** 0.5)
        q = q.repeat_interleave(G, dim=2) * scale   # (B,T,HV,K)
        k = k.repeat_interleave(G, dim=2)           # (B,T,HV,K)

        S = torch.zeros(B, HV, K, V, dtype=torch.float32, device=q.device)
        o = torch.zeros(B, T, HV, V, dtype=torch.float32, device=q.device)
        for t in range(T):
            q_t, k_t, v_t = q[:, t], k[:, t], v[:, t]      # (B,HV,K),(B,HV,K),(B,HV,V)
            gv_t, b_t = gv[:, t], beta[:, t]               # (B,HV,V),(B,HV)
            S = S * gv_t[:, :, None, :].exp()              # decay per V channel
            Sv = (S * v_t[:, :, None, :]).sum(-1)          # S @ v : (B,HV,K)
            k_new = b_t[:, :, None] * (k_t - Sv)           # Oja correction (B,HV,K)
            S = S + k_new[..., None] * v_t[:, :, None, :]  # rank-1 write
            o[:, t] = (S * q_t[..., None]).sum(-2)         # qᵀS : (B,HV,V)
        return o.to(idt)


# --- workload distribution ---
# (B, T, H, HV, K, V). The chunk baseline outputs H heads, so keep HV==H.
WORKLOAD = [
    (1, 128, 4,  4,  64, 64),    # small smoke
    (2, 256, 8,  8, 128, 128),   # medium, real head dim
    (1, 512, 8,  8, 128, 128),   # longer seq
    (2, 256, 4,  4,  64, 64),    # more batch
]

_DEF = (1, 128, 4, 4, 64, 64)


def _rand(shape, device, seed):
    torch.manual_seed(seed)
    B, T, H, HV, K, V = shape
    q = torch.randn(B, T, H, K, device=device) * 0.5
    k = torch.randn(B, T, H, K, device=device) * 0.5
    v = torch.randn(B, T, HV, V, device=device) * 0.5
    # log-space per-V-channel decay, moderate (Oja rule needs decay for stability)
    gv = -torch.rand(B, T, HV, V, device=device) * 0.5 - 0.05
    # Oja update gate kept in [0, 0.5): larger beta with weak decay makes the
    # Hebbian recurrence diverge (state -> 1e15), which is a genuine instability
    # of the Oja rule, not a kernel mismatch.
    beta = torch.rand(B, T, HV, device=device) * 0.5
    return [q, k, v, gv, beta]


def get_inputs():
    return _rand(_DEF, "cpu", 0)


def get_init_inputs():
    return [True, True]  # use_q_l2norm, use_k_l2norm


def make_inputs(shape, device, seed=0):
    return _rand(shape, device, seed)


OUTPUT_INDEX = 0


def baseline(q, k, v, gv, beta, init_args=None):
    """Strong baseline: fla chunk_gated_oja_rule (SOTA Triton, chunkwise)."""
    from fla.ops.gated_oja_rule import chunk_gated_oja_rule
    K = q.shape[-1]
    ql2 = (init_args[0] if init_args else True)
    kl2 = (init_args[1] if init_args and len(init_args) > 1 else True)
    o, _ = chunk_gated_oja_rule(q, k, v, gv, beta, scale=1.0 / (K ** 0.5),
                                output_final_state=False,
                                use_q_l2norm=ql2, use_k_l2norm=kl2)
    return o
