"""K3 task: DeltaNet (delta rule), GVA variant — grouped value attention.

Same error-driven delta-rule state update as K3_delta_rule (no decay gate), but
with GROUPED VALUE ATTENTION: there are H query/key heads and HV = 4*H value
heads. Each q/k head is shared (repeat_interleaved) across `groups = HV/H` value
heads, so the state is [HV, K, V]. Under GVA the head-sharing changes the memory
/ occupancy tradeoff, making it a distinct kernel-research target from the plain
per-head delta rule. There is no native torch op — the chunkwise-scan + recurrent
delta correction must be written by hand.

Math (per v-head, verified against fla.ops.gated_delta_rule.chunk_gated_delta_rule
with g == 0, i.e. no decay == pure delta rule; that op supports GVA whereas
chunk_delta_rule requires H == HV):
    q, k <- L2norm(q), L2norm(k)   # optional, matches use_qk_l2norm_in_kernel
    q    <- q * scale,  scale = 1/sqrt(K)
    (expand q,k from H heads to HV heads via repeat_interleave)
    for t in 0..T:
        delta <- (v_t - Sᵀ k_t) * beta_t   # error-driven correction
        S     <- S + k_t ⊗ delta           # rank-1 state update (no decay)
        o_t   <- Sᵀ q_t                     # readout
    S: recurrent state [K, V] per (batch, v-head)

This Model.forward IS the naive O(T) recurrent reference: correct and
differentiable, but slow (a Python loop over T). It is the CORRECTNESS oracle.

STRONG BASELINE to beat (not naive torch — that is a fake bar):
    fla.ops.gated_delta_rule.chunk_gated_delta_rule with g=0  (SOTA Triton, GVA).
"""
import torch
import torch.nn as nn


def _l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


class Model(nn.Module):
    """Naive O(T) recurrent delta rule with GVA. Correct & differentiable, slow.

    Inputs (batch-first, matching fla convention):
      q, k : (B, T, H,  K)   query/key,  H  key/query heads
      v    : (B, T, HV, V)   value,      HV value heads (HV = H * groups, GVA)
      beta : (B, T, HV)      update gate (already sigmoid'd)
    Output:
      o    : (B, T, HV, V)
    """

    def __init__(self, use_qk_l2norm: bool = True):
        super().__init__()
        self.use_qk_l2norm = use_qk_l2norm

    def forward(self, q, k, v, beta):
        B, T, H, K = q.shape
        HV, V = v.shape[2], v.shape[3]
        groups = HV // H
        idt = q.dtype
        if self.use_qk_l2norm:
            q = _l2norm(q, dim=-1)
            k = _l2norm(k, dim=-1)
        # to (B, HV, T, D) fp32; expand k/q heads to value-head count (GVA)
        q = q.repeat_interleave(groups, dim=2).transpose(1, 2).float()
        k = k.repeat_interleave(groups, dim=2).transpose(1, 2).float()
        v = v.transpose(1, 2).float()
        beta = beta.transpose(1, 2).float()   # (B, HV, T)
        scale = 1.0 / (K ** 0.5)
        q = q * scale

        S = torch.zeros(B, HV, K, V, dtype=torch.float32, device=q.device)
        out = torch.zeros(B, HV, T, V, dtype=torch.float32, device=q.device)
        for t in range(T):
            q_t = q[:, :, t]                    # (B, HV, K)
            k_t = k[:, :, t]                    # (B, HV, K)
            v_t = v[:, :, t]                    # (B, HV, V)
            beta_t = beta[:, :, t][:, :, None]  # (B, HV, 1)
            kv_mem = (S * k_t[:, :, :, None]).sum(dim=-2)   # Sᵀk : (B, HV, V)
            delta = (v_t - kv_mem) * beta_t                 # (B, HV, V)
            S = S + k_t[:, :, :, None] * delta[:, :, None, :]
            out[:, :, t] = (S * q_t[:, :, :, None]).sum(dim=-2)  # Sᵀq
        return out.transpose(1, 2).contiguous().to(idt)  # (B, T, HV, V)


# --- workload distribution --- GVA variant. (B, T, H, HV, K, V), HV = 4*H.
WORKLOAD = [
    (1, 128, 4, 16, 64, 64),
    (2, 256, 8, 32, 128, 128),
    (1, 512, 8, 32, 128, 128),
    (2, 512, 4, 16, 128, 128),
]

_DEF = (1, 128, 4, 16, 64, 64)


def get_inputs():
    torch.manual_seed(0)
    B, T, H, HV, K, V = _DEF
    q = torch.randn(B, T, H, K, dtype=torch.float32) * 0.5
    k = torch.randn(B, T, H, K, dtype=torch.float32) * 0.5
    v = torch.randn(B, T, HV, V, dtype=torch.float32) * 0.5
    beta = torch.rand(B, T, HV, dtype=torch.float32)  # already in (0,1)
    return [q, k, v, beta]


def get_init_inputs():
    return [True]  # use_qk_l2norm


# --- generic-framework contract (make_inputs / baseline) ---
def make_inputs(shape, device, seed=0):
    B, T, H, HV, K, V = shape
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, device=device) * 0.5
    k = torch.randn(B, T, H, K, device=device) * 0.5
    v = torch.randn(B, T, HV, V, device=device) * 0.5
    beta = torch.rand(B, T, HV, device=device)
    return [q, k, v, beta]


OUTPUT_INDEX = 0  # forward returns the output tensor directly


def baseline(q, k, v, beta, init_args=None):
    """Strong baseline: fla chunk_gated_delta_rule with zero decay (== delta rule).

    chunk_delta_rule requires H == HV; the GVA case (HV = 4*H) is handled by
    chunk_gated_delta_rule, which reduces to the plain delta rule when g == 0.
    """
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    K = q.shape[-1]
    l2 = (init_args[0] if init_args else True)
    g = torch.zeros_like(beta)  # no decay -> pure delta rule
    o, _ = chunk_gated_delta_rule(q, k, v, g, beta, scale=1.0 / (K ** 0.5),
                                  output_final_state=False,
                                  use_qk_l2norm_in_kernel=l2)
    return o
