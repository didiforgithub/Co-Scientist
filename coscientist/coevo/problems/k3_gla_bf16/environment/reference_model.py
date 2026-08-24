"""K3 task: GLA (Gated Linear Attention) — chunkwise, per-channel forget gate.

Linear attention where every KEY channel has its own data-dependent forget
gate (log-space). No delta correction — just gated accumulation of k⊗v.
No native torch op: the chunkwise-scan with a per-channel decay must be
written by hand, which is what makes it a genuine kernel-research task.

Math (per head, verified against fla.ops.gla.naive.naive_recurrent_gla):
    q <- q * scale,  scale = 1/sqrt(K)
    for t in 0..T:
        S   <- S * exp(g_t)[:, None] + k_t ⊗ v_t   # g_t decays each K channel
        o_t <- qᵀ_t S                              # readout
    g_t: log-space per-channel gate of shape [K]
    S: recurrent state [K, V] per (batch, head)

This Model.forward IS the naive O(T) recurrent reference: correct and
differentiable, but slow (a Python loop over T). It is the CORRECTNESS oracle.

STRONG BASELINE to beat (not naive torch — that is a fake bar):
    fla.ops.gla.chunk_gla  (SOTA Triton, chunkwise).
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Naive O(T) recurrent GLA. Correct & differentiable, slow.

    Inputs (batch-first, matching fla convention):
      q, k : (B, T, H, K)   query/key
      v    : (B, T, H, V)   value
      g    : (B, T, H, K)   log-space per-channel forget gate (<= 0)
    Output:
      o    : (B, T, H, V)
    """

    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, g):
        B, T, H, K = q.shape
        V = v.shape[-1]
        idt = q.dtype
        q = q.transpose(1, 2).float()
        k = k.transpose(1, 2).float()
        v = v.transpose(1, 2).float()
        g = g.transpose(1, 2).float()          # (B, H, T, K)
        scale = 1.0 / (K ** 0.5)

        S = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
        out = torch.zeros(B, H, T, V, dtype=torch.float32, device=q.device)
        for t in range(T):
            q_t = q[:, :, t] * scale           # (B, H, K)
            k_t = k[:, :, t]                    # (B, H, K)
            v_t = v[:, :, t]                    # (B, H, V)
            g_t = g[:, :, t].exp()             # (B, H, K)
            kv = k_t[:, :, :, None] * v_t[:, :, None, :]     # (B, H, K, V)
            S = S * g_t[:, :, :, None] + kv
            out[:, :, t] = (q_t[:, :, :, None] * S).sum(dim=-2)  # qᵀS
        return out.transpose(1, 2).contiguous().to(idt)  # (B, T, H, V)


# --- workload distribution --- (B, T, H, K, V). chunk=64.
WORKLOAD = [
    (1, 128, 4, 64, 64),
    (2, 256, 8, 128, 128),
    (1, 512, 8, 128, 128),
    (2, 512, 16, 128, 128),
]

_DEF = (1, 128, 4, 64, 64)


def get_inputs():
    torch.manual_seed(0)
    B, T, H, K, V = _DEF
    q = torch.randn(B, T, H, K, dtype=torch.float32) * 0.5
    k = torch.randn(B, T, H, K, dtype=torch.float32) * 0.5
    v = torch.randn(B, T, H, V, dtype=torch.float32) * 0.5
    # log-space per-channel decay, small negative (state slowly decays)
    g = -torch.rand(B, T, H, K, dtype=torch.float32) * 0.1 - 0.01
    return [q, k, v, g]


def get_init_inputs():
    return []


# --- generic-framework contract (make_inputs / baseline) ---
# BF16 variant: inputs are bf16, so tensor-core (mma) utilization in the
# chunkwise matmuls is the optimization target. Tolerance is relaxed to 2e-2.
def make_inputs(shape, device, seed=0):
    B, T, H, K, V = shape
    torch.manual_seed(seed)
    q = (torch.randn(B, T, H, K, device=device) * 0.5).to(torch.bfloat16)
    k = (torch.randn(B, T, H, K, device=device) * 0.5).to(torch.bfloat16)
    v = (torch.randn(B, T, H, V, device=device) * 0.5).to(torch.bfloat16)
    g = (-torch.rand(B, T, H, K, device=device) * 0.1 - 0.01).to(torch.bfloat16)
    return [q, k, v, g]


OUTPUT_INDEX = 0  # forward returns the output tensor directly


def baseline(q, k, v, g, init_args=None):
    """Strong baseline: fla chunk_gla (SOTA Triton)."""
    from fla.ops.gla import chunk_gla
    K = q.shape[-1]
    o, _ = chunk_gla(q, k, v, g, scale=1.0 / (K ** 0.5),
                     output_final_state=False)
    return o
