"""K3 task: Sliding-window causal self-attention (local-attention prefill).

REAL production hotspot: models like Mistral / Gemma / Qwen-long use *sliding
window* attention — each query token attends only to itself and the previous W
keys (a local causal band), not the whole history. This bounds the KV footprint
and the attention cost at long context. The naive dense implementation still
materializes the full S×S score matrix and then throws most of it away with a
banded mask, which is exactly what a good kernel avoids.

This Model.forward IS the naive reference: plain softmax(q·kᵀ/√d)·v with an
explicit banded (causal AND within-window) mask, in fp32. Correct, but slow and
memory-heavy (full S×S scores). It is the CORRECTNESS oracle.

STRONG BASELINE to beat (not naive torch): flashinfer
    single_prefill_with_kv_cache(..., causal=True, window_left=W)
(fused FlashAttention kernel that only walks the local band). Inputs are fp16 to
match the real kernel; the naive ref upcasts to fp32.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Naive sliding-window causal attention. Full S×S banded softmax fp32. Slow.

    Init arg `window_left` W: query i attends keys j with (i-W) <= j <= i.
    """

    def __init__(self, window_left: int = 256):
        super().__init__()
        self.window_left = window_left

    def forward(self, q, k, v):
        # q : (S, H, D) ; k,v : (S, Hkv, D)   NHD single sequence
        S, H, D = q.shape
        Hkv = k.shape[1]
        group = H // Hkv
        W = self.window_left
        scale = 1.0 / (D ** 0.5)
        qf = q.float()
        kf = k.float().repeat_interleave(group, dim=1)   # (S, H, D)
        vf = v.float().repeat_interleave(group, dim=1)
        scores = torch.einsum("qhd,khd->hqk", qf, kf) * scale  # (H, S, S)
        i = torch.arange(S, device=q.device)
        # allowed key j for query i: j <= i (causal) and i - j <= W (window)
        diff = i[:, None] - i[None, :]                   # (S_q, S_k) = i - j
        banded = (diff < 0) | (diff > W)                 # masked-out positions
        scores = scores.masked_fill(banded[None], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("hqk,khd->qhd", probs, vf)    # (S, H, D)
        return out.to(q.dtype)


# --- workload: (seq_len, num_qo_heads, num_kv_heads, head_dim) ---
# GQA H=Hkv*group, head_dim 128. Window is a fixed init arg (WINDOW below) so the
# naive Model and the flashinfer baseline use the exact same band. Longer S with a
# small fixed window => most of the dense S^2 work is wasted (real headroom).
WINDOW = 256

WORKLOAD = [
    (512,  32, 8, 128),   # small smoke  (window ~= seq)
    (1024, 32, 8, 128),   # medium       (window = 1/4 seq)
    (2048, 32, 8, 128),   # long context (window = 1/8 seq)
    (4096, 40, 8, 128),   # very long context (naive S^2 hurts most)
]

_DEF = (512, 32, 8, 128)


def _build(shape, device, seed):
    S, H, Hkv, D = shape
    torch.manual_seed(seed)
    q = (torch.randn(S, H, D, device=device) * 0.5).half()
    k = (torch.randn(S, Hkv, D, device=device) * 0.5).half()
    v = (torch.randn(S, Hkv, D, device=device) * 0.5).half()
    return [q, k, v]


def make_inputs(shape, device, seed=0):
    return _build(shape, device, seed)


def get_inputs():
    return _build(_DEF, "cuda", 0)


def get_init_inputs():
    return [WINDOW]  # window_left (fixed for the whole workload)


OUTPUT_INDEX = 0


def baseline(q, k, v, init_args=None):
    """Strong baseline: flashinfer single_prefill_with_kv_cache (causal + window)."""
    import flashinfer
    W = init_args[0] if init_args else WINDOW
    return flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True, window_left=W)
