"""K3 task: DeltaNet (delta rule) — chunkwise linear attention, no gate.

Linear-attention variant with a *delta rule* state update but NO decay gate:
the state is corrected toward each new value by an error-driven, beta-scaled
rank-1 update. There is no native torch op — the chunkwise-scan + recurrent
delta correction must be written by hand, which is what makes it a genuine
kernel-research task.

Math (per head, verified against fla.ops.delta_rule.naive.delta_rule_recurrence):
    q, k <- L2norm(q), L2norm(k)   # optional, matches use_qk_l2norm_in_kernel
    q    <- q * scale,  scale = 1/sqrt(K)
    for t in 0..T:
        delta <- (v_t - Sᵀ k_t) * beta_t   # error-driven correction
        S     <- S + k_t ⊗ delta           # rank-1 state update (no decay)
        o_t   <- Sᵀ q_t                     # readout
    S: recurrent state [K, V] per (batch, head)

This Model.forward IS the naive O(T) recurrent reference: correct and
differentiable, but slow (a Python loop over T). It is the CORRECTNESS oracle.

STRONG BASELINE to beat (not naive torch — that is a fake bar):
    fla.ops.delta_rule.chunk_delta_rule  (SOTA Triton, chunkwise).
"""
import torch
import torch.nn as nn


def _l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


class Model(nn.Module):
    """Naive O(T) recurrent delta rule. Correct & differentiable, slow.

    Inputs (batch-first, matching fla convention):
      q, k : (B, T, H, K)   query/key
      v    : (B, T, H, V)   value
      beta : (B, T, H)      update gate (already sigmoid'd)
    Output:
      o    : (B, T, H, V)
    """

    def __init__(self, use_qk_l2norm: bool = True):
        super().__init__()
        self.use_qk_l2norm = use_qk_l2norm

    def forward(self, q, k, v, beta):
        B, T, H, K = q.shape
        V = v.shape[-1]
        idt = q.dtype
        if self.use_qk_l2norm:
            q = _l2norm(q, dim=-1)
            k = _l2norm(k, dim=-1)
        # to (B, H, T, D) fp32
        q = q.transpose(1, 2).float()
        k = k.transpose(1, 2).float()
        v = v.transpose(1, 2).float()
        beta = beta.transpose(1, 2).float()   # (B, H, T)
        scale = 1.0 / (K ** 0.5)
        q = q * scale

        S = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
        out = torch.zeros(B, H, T, V, dtype=torch.float32, device=q.device)
        for t in range(T):
            q_t = q[:, :, t]                    # (B, H, K)
            k_t = k[:, :, t]                    # (B, H, K)
            v_t = v[:, :, t]                    # (B, H, V)
            beta_t = beta[:, :, t][:, :, None]  # (B, H, 1)
            kv_mem = (S * k_t[:, :, :, None]).sum(dim=-2)   # Sᵀk : (B, H, V)
            delta = (v_t - kv_mem) * beta_t                 # (B, H, V)
            S = S + k_t[:, :, :, None] * delta[:, :, None, :]
            out[:, :, t] = (S * q_t[:, :, :, None]).sum(dim=-2)  # Sᵀq
        return out.transpose(1, 2).contiguous().to(idt)  # (B, T, H, V)


# --- workload distribution --- (B, T, H, K, V). K=V per shape, chunk=64.
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
    beta = torch.rand(B, T, H, dtype=torch.float32)  # already in (0,1)
    return [q, k, v, beta]


def get_init_inputs():
    return [True]  # use_qk_l2norm


# --- generic-framework contract (make_inputs / baseline) ---
def make_inputs(shape, device, seed=0):
    B, T, H, K, V = shape
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, device=device) * 0.5
    k = torch.randn(B, T, H, K, device=device) * 0.5
    v = torch.randn(B, T, H, V, device=device) * 0.5
    beta = torch.rand(B, T, H, device=device)
    return [q, k, v, beta]


OUTPUT_INDEX = 0  # forward returns the output tensor directly


def baseline(q, k, v, beta, init_args=None):
    """Strong baseline: fla chunk_delta_rule (SOTA Triton).

    NOTE: chunk_delta_rule asserts inputs are NOT float32 — cast to bf16.
    """
    from fla.ops.delta_rule import chunk_delta_rule
    K = q.shape[-1]
    l2 = (init_args[0] if init_args else True)
    qb, kb, vb, bb = (x.to(torch.bfloat16) for x in (q, k, v, beta))
    o, _ = chunk_delta_rule(qb, kb, vb, bb, scale=1.0 / (K ** 0.5),
                            output_final_state=False, use_qk_l2norm_in_kernel=l2)
    return o.to(q.dtype)
