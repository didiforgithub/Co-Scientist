"""K3 task: RWKV-7 — DPLR (diagonal-plus-low-rank) delta-rule linear attention.

RWKV-7 ("Goose") generalizes the delta rule to a diagonal-plus-low-rank state
transition: each step the state is decayed per-channel (diagonal `w`) AND
corrected by a rank-1 term built from two learned low-rank vectors `a`,`b`
(the "in-context learning" removal/replacement keys). There is NO native torch
op — the chunkwise DPLR scan (fla routes chunk_rwkv7 -> chunk_dplr_delta_rule)
must be hand-written.

Math (per head, verified against
fla.ops.generalized_delta_rule.dplr.naive.dplr_recurrence, scale=1.0):
    for t in 0..T:
        kv  <- k_t ⊗ v_t + ((S · a_t) summed over K) ⊗ b_t   # rank-1 DPLR correction
        S   <- S * exp(w_t)[:,None] + kv                      # per-K-channel diag decay
        o_t <- rᵀ_t S                                          # readout (r = query)
    w_t: log-space per-channel decay [K];  a_t,b_t: low-rank vectors [K]
    S: recurrent state [K, V] per (batch, head)

This Model.forward IS the naive O(T) recurrent oracle: correct, differentiable, slow.

STRONG BASELINE to beat: fla.ops.rwkv7.chunk_rwkv7  (SOTA Triton, chunkwise DPLR).
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Naive O(T) recurrent RWKV-7 DPLR delta rule. Correct & differentiable, slow.

    Inputs (batch-first, matching fla.ops.rwkv7 convention):
      r    : (B, T, H, K)   receptance / query
      w    : (B, T, H, K)   log-space per-channel decay (<= 0)
      k    : (B, T, H, K)   key
      v    : (B, T, H, V)   value
      a, b : (B, T, H, K)   low-rank DPLR correction vectors
    Output:
      o    : (B, T, H, V)
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, r, w, k, v, a, b):
        B, T, H, K = r.shape
        V = v.shape[-1]
        idt = r.dtype
        r, w, k, v, a, b = (x.transpose(1, 2).float() for x in (r, w, k, v, a, b))
        scale = self.scale

        S = torch.zeros(B, H, K, V, dtype=torch.float32, device=r.device)
        o = torch.zeros(B, H, T, V, dtype=torch.float32, device=r.device)
        for i in range(T):
            r_i = r[:, :, i]                          # (B,H,K)
            k_i = k[:, :, i]                          # (B,H,K)
            v_i = v[:, :, i]                          # (B,H,V)
            a_i = a[:, :, i]                          # (B,H,K)
            b_i = b[:, :, i]                          # (B,H,K)
            # DPLR correction: k⊗v + ((S·a) over K) ⊗ b
            kv = (k_i[..., None] * v_i[..., None, :]
                  + (S * a_i[..., None]).sum(-2, keepdim=True) * b_i[..., None])
            S = S * w[:, :, i].exp()[..., None] + kv
            o[:, :, i] = torch.einsum('bhd,bhdm->bhm', r_i * scale, S)
        return o.transpose(1, 2).contiguous().to(idt)  # (B, T, H, V)


# --- workload distribution --- BIG-DIM variant. (B, T, H, K, V). chunk=16.
# Large head dim K=V=256: the wide DPLR chunk matmul (256x256 state per head) is
# the optimization target. NOTE: rwkv7's chunk kernel does not support bf16-stable
# alignment (fla warns fp32 is the only accurate path), so this variant stays fp32
# and instead grows the head dim; decay is strengthened so the wide state stays
# contractive and the naive oracle aligns with the chunk kernel.
WORKLOAD = [
    (1, 256, 4, 256, 256),
    (2, 256, 8, 256, 256),
    (1, 512, 4, 256, 256),
]

_DEF = (1, 256, 4, 256, 256)


def _rand(shape, device, seed):
    torch.manual_seed(seed)
    B, T, H, K, V = shape
    r = torch.randn(B, T, H, K, device=device) * 0.5
    k = torch.randn(B, T, H, K, device=device) * 0.5
    v = torch.randn(B, T, H, V, device=device) * 0.5
    # w: log-space per-channel decay. The wide (256x256) state needs a stronger
    # contraction than the base task for the chunk kernel to track the naive
    # recurrence, so w in [-0.6, -0.1].
    w = -torch.rand(B, T, H, K, device=device) * 0.5 - 0.1
    # DPLR low-rank vectors: a is an L2-normalized removal key, b = -a * lr (small),
    # matching RWKV-7's constraint a·b in (-1, 0] that keeps the transition contractive.
    a = torch.randn(B, T, H, K, device=device)
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-6)
    b = -a * torch.rand(B, T, H, 1, device=device)
    return [r, w, k, v, a, b]


def get_inputs():
    return _rand(_DEF, "cpu", 0)


def get_init_inputs():
    return [1.0]  # scale


def make_inputs(shape, device, seed=0):
    return _rand(shape, device, seed)


OUTPUT_INDEX = 0


def baseline(r, w, k, v, a, b, init_args=None):
    """Strong baseline: fla chunk_rwkv7 (SOTA Triton, chunkwise DPLR)."""
    from fla.ops.rwkv7 import chunk_rwkv7
    scale = (init_args[0] if init_args else 1.0)
    o, _ = chunk_rwkv7(r, w, k, v, a, b, scale=scale, output_final_state=False)
    return o
