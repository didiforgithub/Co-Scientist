"""K3 task: MesaNet — test-time regression (mesa-optimization) linear attention.

MesaNet replaces the delta-rule's single gradient step with an *exact* least-
squares solve at every position: the readout solves (H_kk + diag(lamb)) x = q,
then reads o = x·H_kv, where H_kk = sum of decayed β·k⊗k and H_kv = sum of
decayed β·k⊗v. The fla Triton kernel approximates the solve with `max_CG_iteration`
conjugate-gradient steps. There is NO native torch op — the chunkwise scan plus
in-kernel CG solve is a genuinely hard hand-written kernel.

Math (per head, verified against fla.ops.mesa_net.naive.naive_mesa_net_exact):
    for t in 0..T:
        H_kk <- H_kk*exp(g_t) + (beta_t*k_t) ⊗ k_t     # [K,K]
        H_kv <- H_kv*exp(g_t) + (beta_t*k_t) ⊗ v_t     # [K,V]
    x_t  = solve(H_kk(t) + diag(lamb), q_t)             # ridge-regularized solve
    o_t  = x_t · H_kv(t)                                # readout
    g_t: head-wise log decay;  beta_t: head-wise gate;  lamb: [H,K] ridge (>=0.25)

This Model.forward IS the exact O(T) oracle (linalg.solve), correct & slow.
Note: the baseline uses 30 CG iterations, so agreement is at CG-convergence tol.

STRONG BASELINE to beat: fla.ops.mesa_net.chunk_mesa_net  (SOTA Triton + in-kernel CG).
"""
import torch
import torch.nn as nn


def _l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


class Model(nn.Module):
    """Exact O(T) MesaNet oracle (per-step ridge least-squares solve). Slow.

    Inputs (batch-first, matching fla.ops.mesa_net convention):
      q, k : (B, T, H, K)   query / key (L2-normalized internally)
      v    : (B, T, H, V)   value  (V == K)
      g    : (B, T, H)      head-wise log decay (<= 0)
      beta : (B, T, H)      head-wise update gate
      lamb : (H, K)         per-(head,channel) ridge regularizer (>= 0.25)
    Output:
      o    : (B, T, H, V)
    """

    def __init__(self, use_qk_l2norm: bool = True):
        super().__init__()
        self.use_qk_l2norm = use_qk_l2norm

    def forward(self, q, k, v, g, beta, lamb):
        idt = q.dtype
        if self.use_qk_l2norm:
            q = _l2norm(q, dim=-1)
            k = _l2norm(k, dim=-1)
        B, L, H, d = q.shape
        q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
        lamb = lamb.float()

        h_kk = torch.zeros(B, H, d, d, device=q.device)
        h_kv = torch.zeros(B, H, d, d, device=q.device)
        h_kk_all = torch.zeros(B, L, H, d, d, device=q.device)
        h_kv_all = torch.zeros(B, L, H, d, d, device=q.device)
        for i in range(L):
            decay = g[:, i, :, None, None].exp()
            bk = (k[:, i] * beta[:, i, :, None])[..., None]   # (B,H,d,1)
            h_kk = h_kk * decay + bk * k[:, i, :, None, :]
            h_kv = h_kv * decay + bk * v[:, i, :, None, :]
            h_kk_all[:, i] = h_kk
            h_kv_all[:, i] = h_kv

        # exact ridge solve: x = (H_kk + diag(lamb))^{-1} q ;  o = x · H_kv
        A = h_kk_all + torch.diag_embed(lamb)[None, None, ...]
        x = torch.linalg.solve(A, q)
        o = (x[..., :, None] * h_kv_all).sum(-2)
        return o.to(idt)


# --- workload distribution --- (B, T, H, K, V). K==V (square state). chunk=64.
WORKLOAD = [
    (1, 128, 4, 128, 128),
    (2, 256, 4, 128, 128),
    (1, 512, 4, 128, 128),
]

_DEF = (1, 128, 4, 128, 128)


def _rand(shape, device, seed):
    torch.manual_seed(seed)
    B, T, H, K, V = shape
    q = torch.randn(B, T, H, K, device=device) * 0.5
    k = torch.randn(B, T, H, K, device=device) * 0.5
    v = torch.randn(B, T, H, V, device=device) * 0.5
    # head-wise log decay (logsigmoid -> negative), gate in (0,1)
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, device=device))
    beta = torch.rand(B, T, H, device=device)
    # ridge lamb per (head, channel), lower bound 0.25 for numerical stability
    lamb = torch.nn.functional.softplus(torch.randn(H, K, device=device)) + 0.25
    return [q, k, v, g, beta, lamb]


def get_inputs():
    return _rand(_DEF, "cpu", 0)


def get_init_inputs():
    return [True]  # use_qk_l2norm


def make_inputs(shape, device, seed=0):
    return _rand(shape, device, seed)


OUTPUT_INDEX = 0


def baseline(q, k, v, g, beta, lamb, init_args=None):
    """Strong baseline: fla chunk_mesa_net (SOTA Triton + in-kernel CG solve)."""
    from fla.ops.mesa_net import chunk_mesa_net
    l2 = (init_args[0] if init_args else True)
    o, _, _ = chunk_mesa_net(q, k, v, g, beta, lamb,
                             output_final_state=False,
                             max_CG_iteration=30,
                             use_qk_l2norm_in_kernel=l2)
    return o
