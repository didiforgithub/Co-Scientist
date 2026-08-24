"""K3 task: RWKV-6 — chunkwise linear attention with log-decay + bonus.

RWKV-6 receptance-weighted key-value: like GLA (per-channel log-space decay
`w` on the state) but with an extra "bonus" term `u` that boosts the CURRENT
timestep's contribution before it is folded into the decaying state. The
readout mixes the pre-update state with the u-boosted current k⊗v. No native
torch op: the chunkwise-scan with decay + bonus must be written by hand, which
is what makes it a genuine kernel-research task.

Math (per head, verified against fla.ops.rwkv6.recurrent_naive.naive_recurrent_rwkv6):
    r <- r * scale,  scale = 1/sqrt(K)          # r is the "query"/receptance
    for t in 0..T:
        kv_t <- k_t ⊗ v_t                        # (K, V)
        o_t  <- rᵀ_t (S + u ⊙ kv_t)              # bonus u boosts current step
        S    <- S * exp(w_t)[:, None] + kv_t     # per-channel log decay w_t
    w_t: log-space per-channel decay [K];  u: per-head bonus [K]
    S: recurrent state [K, V] per (batch, head)

This Model.forward IS the naive O(T) recurrent reference: correct and
differentiable, but slow (a Python loop over T). It is the CORRECTNESS oracle.

STRONG BASELINE to beat (not naive torch — that is a fake bar):
    fla.ops.rwkv6.chunk_rwkv6  (SOTA Triton, chunkwise).
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Naive O(T) recurrent RWKV-6. Correct & differentiable, slow.

    Inputs (batch-first, matching fla convention):
      r, k : (B, T, H, K)   receptance(query)/key
      v    : (B, T, H, V)   value
      w    : (B, T, H, K)   log-space per-channel decay (<= 0)
      u    : (H, K)         per-head bonus (data-independent parameter)
    Output:
      o    : (B, T, H, V)
    """

    def __init__(self):
        super().__init__()

    def forward(self, r, k, v, w, u):
        B, T, H, K = r.shape
        V = v.shape[-1]
        idt = r.dtype
        r = r.transpose(1, 2).float()
        k = k.transpose(1, 2).float()
        v = v.transpose(1, 2).float()
        w = w.transpose(1, 2).float()          # (B, H, T, K)
        u = u.float()                          # (H, K)
        scale = 1.0 / (K ** 0.5)

        S = torch.zeros(B, H, K, V, dtype=torch.float32, device=r.device)
        out = torch.zeros(B, H, T, V, dtype=torch.float32, device=r.device)
        u_e = u[None, :, :, None]              # (1, H, K, 1)
        for t in range(T):
            r_t = r[:, :, t] * scale           # (B, H, K)
            k_t = k[:, :, t]                   # (B, H, K)
            v_t = v[:, :, t]                   # (B, H, V)
            w_t = w[:, :, t].exp()             # (B, H, K)
            kv = k_t[:, :, :, None] * v_t[:, :, None, :]     # (B, H, K, V)
            o_t = (S + u_e * kv) * r_t[:, :, :, None]        # (B, H, K, V)
            out[:, :, t] = o_t.sum(dim=-2)
            S = S * w_t[:, :, :, None] + kv
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
    r = torch.randn(B, T, H, K, dtype=torch.float32) * 0.5
    k = torch.randn(B, T, H, K, dtype=torch.float32) * 0.5
    v = torch.randn(B, T, H, V, dtype=torch.float32) * 0.5
    # log-space per-channel decay, small negative
    w = -torch.rand(B, T, H, K, dtype=torch.float32) * 0.1 - 0.01
    u = torch.randn(H, K, dtype=torch.float32) * 0.5
    return [r, k, v, w, u]


def get_init_inputs():
    return []


# --- generic-framework contract (make_inputs / baseline) ---
def make_inputs(shape, device, seed=0):
    B, T, H, K, V = shape
    torch.manual_seed(seed)
    r = torch.randn(B, T, H, K, device=device) * 0.5
    k = torch.randn(B, T, H, K, device=device) * 0.5
    v = torch.randn(B, T, H, V, device=device) * 0.5
    w = -torch.rand(B, T, H, K, device=device) * 0.1 - 0.01
    u = torch.randn(H, K, device=device) * 0.5
    return [r, k, v, w, u]


OUTPUT_INDEX = 0  # forward returns the output tensor directly


def baseline(r, k, v, w, u, init_args=None):
    """Strong baseline: fla chunk_rwkv6 (SOTA Triton).

    NOTE: the rwkv6 chunk kernel hits a misaligned-address CUDA fault on fp32
    inputs; it must be run in bf16 (r/k/v/w cast; u stays fp32 per fla API).
    """
    from fla.ops.rwkv6 import chunk_rwkv6
    K = r.shape[-1]
    rb, kb, vb, wb = (x.to(torch.bfloat16) for x in (r, k, v, w))
    o, _ = chunk_rwkv6(rb, kb, vb, wb, u, scale=1.0 / (K ** 0.5),
                       output_final_state=False)
    return o.to(r.dtype)
