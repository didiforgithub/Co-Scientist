"""K3 task: Ragged (variable-length) causal prefill attention with strong GQA.

REAL production hotspot: LLM prefill / training packs many prompts of different
lengths into one flat token buffer described by cumulative sequence-length offsets
(cu_seqlens). Each prompt does full causal self-attention over only its own tokens.
Modern models (Llama-3-70B, Qwen) use heavy grouped-query attention (GQA): far
fewer KV heads than query heads (group = 8), so each KV head is shared by 8 query
heads. This is the varlen FlashAttention path with high group ratio.

This Model.forward IS the naive reference: it slices out each sequence with the
cu_seqlens offsets, repeat-interleaves the KV heads up to the query heads, and runs
a plain causal softmax(q·kᵀ/√d)·v in fp32 per sequence (python loop over batch,
full materialized scores). Correct, but slow. It is the CORRECTNESS oracle.

STRONG BASELINE to beat (not naive torch): flashinfer
    BatchPrefillWithRaggedKVCacheWrapper  (fused varlen FlashAttention kernel).
Inputs are fp16 to match the real kernel; the naive ref upcasts to fp32.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Naive ragged causal prefill with GQA expand. Per-seq full causal softmax fp32."""

    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, cu_seqlens):
        # q : (total_tokens, H, D) ; k,v : (total_tokens, Hkv, D)   ragged-packed NHD
        # cu_seqlens : (B+1,) int32  cumulative sequence lengths
        H, D = q.shape[1], q.shape[2]
        Hkv = k.shape[1]
        group = H // Hkv
        scale = 1.0 / (D ** 0.5)
        total = q.shape[0]
        out = torch.empty(total, H, D, dtype=torch.float32, device=q.device)
        B = cu_seqlens.numel() - 1
        for b in range(B):
            s, e = int(cu_seqlens[b]), int(cu_seqlens[b + 1])
            S = e - s
            qb = q[s:e].float()                       # (S, H, D)
            kb = k[s:e].float().repeat_interleave(group, dim=1)  # (S, H, D) GQA expand
            vb = v[s:e].float().repeat_interleave(group, dim=1)
            scores = torch.einsum("qhd,khd->hqk", qb, kb) * scale  # (H, S, S)
            causal = torch.triu(torch.ones(S, S, device=q.device, dtype=torch.bool), 1)
            scores = scores.masked_fill(causal[None], float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out[s:e] = torch.einsum("hqk,khd->qhd", probs, vb)  # (S, H, D)
        return out.to(q.dtype)


# --- workload: (seqlens_tuple, num_qo_heads, num_kv_heads, head_dim) ---
# heavy GQA (group = 8): H = 8*Hkv. head_dim 128.
WORKLOAD = [
    ((64, 128, 96, 200), 64, 8, 128),                    # small smoke, group=8
    ((256, 384, 128, 512), 64, 8, 128),                  # medium
    ((512, 768, 256, 1024, 640), 64, 8, 128),            # long, 5 seqs
    ((1024, 1536, 768, 2048, 512, 900), 64, 8, 128),     # large, 6 seqs
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
    return []  # Model is shape-agnostic; head dims come from the input tensors


OUTPUT_INDEX = 0

_WS = {}


def baseline(q, k, v, cu_seqlens, init_args=None):
    """Strong baseline: flashinfer BatchPrefillWithRaggedKVCacheWrapper (GQA causal)."""
    import flashinfer
    H, D = q.shape[1], q.shape[2]
    Hkv = k.shape[1]
    dev = q.device
    if dev not in _WS:
        _WS[dev] = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    wrap = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(_WS[dev], "NHD")
    wrap.plan(cu_seqlens, cu_seqlens, H, Hkv, D, causal=True,
              q_data_type=q.dtype, kv_data_type=k.dtype)
    return wrap.run(q, k, v)  # (total, H, D)
