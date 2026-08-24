"""K3 task: Ragged (variable-length) NON-causal bidirectional prefill with GQA.

REAL production hotspot: encoder-style / bidirectional attention (embedding models,
prefix-LM, cross-attention encoders) packs many variable-length sequences into one
flat buffer (cu_seqlens) and every token attends to all tokens of its own sequence
(no causal mask). Heavy GQA (group = 8) is common. This is the varlen full-attention
path — the counterpart to the causal ragged prefill.

This Model.forward IS the naive reference: per-sequence full (non-causal) softmax
attention with KV heads repeat-interleaved to query heads, in fp32. Correct, slow.

STRONG BASELINE to beat: flashinfer BatchPrefillWithRaggedKVCacheWrapper(causal=False).
Inputs are fp16 to match the real kernel; the naive ref upcasts to fp32.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Naive ragged bidirectional prefill with GQA expand. Full softmax fp32 per seq."""

    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, cu_seqlens):
        H, D = q.shape[1], q.shape[2]
        Hkv = k.shape[1]
        group = H // Hkv
        scale = 1.0 / (D ** 0.5)
        total = q.shape[0]
        out = torch.empty(total, H, D, dtype=torch.float32, device=q.device)
        B = cu_seqlens.numel() - 1
        for b in range(B):
            s, e = int(cu_seqlens[b]), int(cu_seqlens[b + 1])
            qb = q[s:e].float()
            kb = k[s:e].float().repeat_interleave(group, dim=1)
            vb = v[s:e].float().repeat_interleave(group, dim=1)
            scores = torch.einsum("qhd,khd->hqk", qb, kb) * scale  # (H, S, S)
            probs = torch.softmax(scores, dim=-1)                  # no mask
            out[s:e] = torch.einsum("hqk,khd->qhd", probs, vb)
        return out.to(q.dtype)


WORKLOAD = [
    ((64, 128, 96, 200), 64, 8, 128),
    ((256, 384, 128, 512), 64, 8, 128),
    ((512, 768, 256, 1024, 640), 64, 8, 128),
    ((1024, 1536, 768, 2048, 512, 900), 64, 8, 128),
]

_DEF = ((64, 128, 96, 200), 64, 8, 128)


def _build_ragged(shape, device, seed):
    seqlens, H, Hkv, D = shape
    torch.manual_seed(seed)
    cu = torch.zeros(len(seqlens) + 1, device=device, dtype=torch.int32)
    cu[1:] = torch.tensor(seqlens, device=device, dtype=torch.int32).cumsum(0)
    total = int(cu[-1])
    q = (torch.randn(total, H, D, device=device) * 0.5).half()
    k = (torch.randn(total, Hkv, D, device=device) * 0.5).half()
    v = (torch.randn(total, Hkv, D, device=device) * 0.5).half()
    return [q, k, v, cu]


def make_inputs(shape, device, seed=0):
    return _build_ragged(shape, device, seed)


def get_inputs():
    return _build_ragged(_DEF, "cuda", 0)


def get_init_inputs():
    return []


OUTPUT_INDEX = 0

_WS = {}


def baseline(q, k, v, cu_seqlens, init_args=None):
    """Strong baseline: flashinfer BatchPrefillWithRaggedKVCacheWrapper (non-causal)."""
    import flashinfer
    H, D = q.shape[1], q.shape[2]
    Hkv = k.shape[1]
    dev = q.device
    if dev not in _WS:
        _WS[dev] = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    wrap = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(_WS[dev], "NHD")
    wrap.plan(cu_seqlens, cu_seqlens, H, Hkv, D, causal=False,
              q_data_type=q.dtype, kv_data_type=k.dtype)
    return wrap.run(q, k, v)
