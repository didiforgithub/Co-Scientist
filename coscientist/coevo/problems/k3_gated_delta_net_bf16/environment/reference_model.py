"""K3 task: Gated DeltaNet (GDN) — chunkwise linear attention with recurrent state.

REAL production hotspot: GDN layers are 75% of the layers in our Qwen3.6-27B /
3.5-35B-A3B models. Training goes through transformers' Qwen3NextGatedDeltaNet,
which calls the `fla` (flash-linear-attention) Triton kernels. There is NO native
torch op for this — the chunkwise-scan + recurrent-state update must be written by
hand, which is exactly why it is a genuine kernel-research task (unlike attention
or fused-loss, where cuBLAS/torch is already near-optimal).

Math (per v-head, gated delta rule, verified against transformers'
`torch_recurrent_gated_delta_rule`, modeling_qwen3_next.py:454):
    q, k <- L2norm(q), L2norm(k)          # optional, on by default in Qwen3Next
    q <- q * scale,  scale = 1/sqrt(K)
    for t in 0..T:
        S <- S * exp(g_t)                 # state decay, g_t log-space scalar per head
        delta <- (v_t - Sᵀ k_t) * beta_t  # delta rule correction
        S <- S + k_t ⊗ delta              # rank-1 state update
        o_t <- Sᵀ q_t                      # readout
    S: recurrent state [K, V] per (batch, v-head)

This Model.forward IS the naive O(T) recurrent reference: correct and
differentiable, but slow (a Python loop over T). It is the CORRECTNESS oracle.

STRONG BASELINE to beat (not naive torch — that is a fake bar):
    fla.ops.gated_delta_rule.chunk_gated_delta_rule  (SOTA Triton, chunkwise).
`ModelNew` must match forward output AND backward grads (dq,dk,dv,dg,dbeta),
across a workload of shapes, staying within a memory budget, faster than... itself
vs the naive ref; the real target is to approach fla's chunkwise throughput.
"""
import torch
import torch.nn as nn


def _l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


class Model(nn.Module):
    """Naive O(T) recurrent gated delta rule. Correct & differentiable, slow.

    Inputs (batch-first, matching Qwen3Next / fla convention):
      q, k : (B, T, H,  K)   query/key,  H  key/query heads
      v    : (B, T, HV, V)   value,      HV value heads (HV = H * groups, GVA)
      g    : (B, T, HV)      log-space decay (already = -exp(A_log)*softplus(...))
      beta : (B, T, HV)      update gate (already sigmoid'd)
    Output:
      o    : (B, T, HV, V)
    """

    def __init__(self, use_qk_l2norm: bool = True):
        super().__init__()
        self.use_qk_l2norm = use_qk_l2norm

    def forward(self, q, k, v, g, beta):
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
        g = g.transpose(1, 2).float()          # (B, HV, T)
        beta = beta.transpose(1, 2).float()    # (B, HV, T)
        scale = 1.0 / (K ** 0.5)
        q = q * scale

        S = torch.zeros(B, HV, K, V, dtype=torch.float32, device=q.device)
        out = torch.zeros(B, HV, T, V, dtype=torch.float32, device=q.device)
        for t in range(T):
            q_t = q[:, :, t]                    # (B, HV, K)
            k_t = k[:, :, t]                    # (B, HV, K)
            v_t = v[:, :, t]                    # (B, HV, V)
            g_t = g[:, :, t].exp()[:, :, None, None]   # (B, HV, 1, 1)
            beta_t = beta[:, :, t][:, :, None]         # (B, HV, 1)
            S = S * g_t
            kv_mem = (S * k_t[:, :, :, None]).sum(dim=-2)   # Sᵀk : (B, HV, V)
            delta = (v_t - kv_mem) * beta_t                 # (B, HV, V)
            S = S + k_t[:, :, :, None] * delta[:, :, None, :]
            out[:, :, t] = (S * q_t[:, :, :, None]).sum(dim=-2)  # Sᵀq
        return out.transpose(1, 2).contiguous().to(idt)  # (B, T, HV, V)


# --- workload distribution (real shapes from Qwen3.6-27B / 3.5-35B GDN layers) ---
# (B, T, H, HV, K, V). H=key/query heads, HV=value heads (GVA), K=V=128, chunk=64.
# Short T first (fast inner-loop, the naive ref is O(T)); larger T for real scale.
WORKLOAD = [
    (1,  128, 4,  8,  64, 64),    # small smoke (naive loop over 128 steps ok)
    (2,  256, 8, 16, 128, 128),   # medium, real head dim
    (1,  512, 16, 48, 128, 128),  # Qwen3.6-27B head config, seq 512
    (2,  512, 16, 32, 128, 128),  # Qwen3.5-35B head config
]

_DEF = (1, 128, 4, 8, 64, 64)


def get_inputs():
    torch.manual_seed(0)
    B, T, H, HV, K, V = _DEF
    q = torch.randn(B, T, H, K, dtype=torch.float32) * 0.5
    k = torch.randn(B, T, H, K, dtype=torch.float32) * 0.5
    v = torch.randn(B, T, HV, V, dtype=torch.float32) * 0.5
    # g log-space decay: small negative (state slowly decays), realistic range
    g = -torch.rand(B, T, HV, dtype=torch.float32) * 0.1 - 0.01
    beta = torch.rand(B, T, HV, dtype=torch.float32)  # already in (0,1)
    return [q, k, v, g, beta]


def get_init_inputs():
    return [True]  # use_qk_l2norm


# --- generic-framework contract (make_inputs / baseline) ---
# BF16 variant: inputs are bf16, so tensor-core (mma) utilization in the
# chunkwise matmuls is the optimization target. Tolerance is relaxed to 2e-2.
def make_inputs(shape, device, seed=0):
    B, T, H, HV, K, V = shape
    torch.manual_seed(seed)
    q = (torch.randn(B, T, H, K, device=device) * 0.5).to(torch.bfloat16)
    k = (torch.randn(B, T, H, K, device=device) * 0.5).to(torch.bfloat16)
    v = (torch.randn(B, T, HV, V, device=device) * 0.5).to(torch.bfloat16)
    g = (-torch.rand(B, T, HV, device=device) * 0.1 - 0.01).to(torch.bfloat16)
    beta = torch.rand(B, T, HV, device=device).to(torch.bfloat16)
    return [q, k, v, g, beta]


OUTPUT_INDEX = 0  # forward returns the output tensor directly


def baseline(q, k, v, g, beta, init_args=None):
    """Strong baseline: fla chunk_gated_delta_rule (SOTA Triton)."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    K = q.shape[-1]
    l2 = (init_args[0] if init_args else True)
    o, _ = chunk_gated_delta_rule(q, k, v, g, beta, scale=1.0 / (K ** 0.5),
                                  output_final_state=False, use_qk_l2norm_in_kernel=l2)
    return o
