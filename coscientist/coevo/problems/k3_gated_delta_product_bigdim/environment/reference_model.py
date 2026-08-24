"""K3 task: Gated DeltaProduct — multi-householder gated delta rule.

DeltaProduct applies MULTIPLE (num_householder) delta-rule state transitions per
time step, i.e. a product of Householder-like rank-1 updates, giving a richer
per-token state transition than a single delta update. There is NO native torch
op — the chunkwise scan over the expanded (T * num_householder) key/value stream
is a genuine hand-written kernel.

Math (per head, on matrix state h in R^{K x V}, verified against
fla.ops.gated_delta_product.naive.naive_recurrent_gated_delta_product):
    scale = 1/sqrt(K),  q <- q * scale
    for t in 0..T:
        h <- exp(g_t) * h                                  # head-wise log decay
        for j in 0..num_householder:                       # product of updates
            idx = t*num_householder + j
            h <- h + (v_idx - (k_idx · h)) * beta_idx  ⊗ k_idx   # delta update
        o_t <- (q_t · h)                                   # readout
    h: R^{K x V} recurrent state per (batch, head)

`g` is the head-wise log-decay applied ONCE per real time step [B,T,H]. `k`, `v`,
`beta` live on the expanded axis of length T*num_householder. This Model.forward IS
the naive O(T * num_householder) recurrent oracle.

NOTE: fla's chunk kernel requires bf16 inputs (float32 unsupported), so the
workload uses bfloat16; allclose tolerance is the standard 1e-2.

STRONG BASELINE to beat: fla.ops.gated_delta_product.chunk_gated_delta_product.
"""
import torch
import torch.nn as nn


def _l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


class Model(nn.Module):
    """Naive O(T*nh) multi-householder gated delta recurrence. Slow but correct.

    Inputs (batch-first):
      q    : (B, T,    H, K)   query
      k    : (B, T*nh, H, K)   keys on the expanded householder axis
      v    : (B, T*nh, H, V)   values on the expanded householder axis
      g    : (B, T,    H)      head-wise log-decay (once per real step)
      beta : (B, T*nh, H)      update gates on the expanded axis
    Output:
      o    : (B, T, H, V)
    """

    def __init__(self, num_householder: int = 2):
        super().__init__()
        self.num_householder = num_householder

    def forward(self, q, k, v, g, beta):
        nh = self.num_householder
        idt = q.dtype
        B, T, H, K = q.shape
        V = v.shape[-1]
        scale = K ** -0.5
        q, k, v, beta, g = (x.float() for x in (q, k, v, beta, g))
        # L2-normalize q,k (matches use_qk_l2norm_in_kernel=True; keeps the
        # multi-householder delta state well conditioned).
        q = _l2norm(q, dim=-1)
        k = _l2norm(k, dim=-1)
        q = q * scale
        h = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
        o = torch.zeros(B, T, H, V, dtype=torch.float32, device=q.device)
        for i in range(T):
            h = h * g[:, i, :].exp()[..., None, None]          # (B,H,K,V)
            for j in range(nh):
                idx = i * nh + j
                k_ij = k[:, idx, :, :]                          # (B,H,K)
                v_ij = v[:, idx, :, :]                          # (B,H,V)
                beta_ij = beta[:, idx, :]                       # (B,H)
                delta = (v_ij - (h * k_ij[..., None]).sum(-2))  # (B,H,V)
                h = h + delta.unsqueeze(-2) * k_ij[..., None] * beta_ij[..., None, None]
            q_i = q[:, i, :, :]                                 # (B,H,K)
            o[:, i] = (h * q_i[..., None]).sum(-2)              # (B,H,V)
        return o.to(idt)


# --- workload distribution --- BIG-DIM variant. (B, T, H, K, V, nh).
# Large head dim K=V=256 (nh=2): the wide delta-product state (256x256 per head)
# stresses the chunkwise matmul. bf16 inputs (chunk kernel requires it).
WORKLOAD = [
    (1, 256, 4, 256, 256, 2),
    (2, 256, 8, 256, 256, 2),
    (1, 512, 4, 256, 256, 2),
]
_DEF_D = (1, 256, 4, 256, 256, 2)

_DEF = _DEF_D


def _rand(shape, device, seed):
    torch.manual_seed(seed)
    B, T, H, K, V, nh = shape
    dt = torch.bfloat16
    q = (torch.randn(B, T, H, K, device=device) * 0.5).to(dt)
    k = (torch.randn(B, T * nh, H, K, device=device) * 0.5).to(dt)
    v = (torch.randn(B, T * nh, H, V, device=device) * 0.5).to(dt)
    g = torch.nn.functional.logsigmoid(torch.rand(B, T, H, device=device)).to(dt)
    beta = torch.rand(B, T * nh, H, device=device).to(dt)   # gate in [0,1)
    return [q, k, v, g, beta]


def get_inputs():
    return _rand(_DEF, "cpu", 0)


def get_init_inputs():
    return [_DEF[-1]]   # num_householder


def make_inputs(shape, device, seed=0):
    return _rand(shape, device, seed)


OUTPUT_INDEX = 0


def baseline(q, k, v, g, beta, init_args=None):
    """Strong baseline: fla chunk_gated_delta_product (SOTA Triton)."""
    from fla.ops.gated_delta_product import chunk_gated_delta_product
    K = q.shape[-1]
    nh = k.shape[1] // q.shape[1]
    o, _ = chunk_gated_delta_product(q, k, v, g, beta, num_householder=nh,
                                     scale=K ** -0.5, output_final_state=False,
                                     use_qk_l2norm_in_kernel=True)
    return o
