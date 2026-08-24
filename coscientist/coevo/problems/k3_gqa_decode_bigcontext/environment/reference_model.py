"""K3 task: Long-context paged GQA decode attention (memory-bound decode stress).

REAL production hotspot: at long context (many thousands of cached tokens) the
autoregressive decode step is fully memory-bandwidth bound — every new token must
read the entire paged KV history to produce one output token. Aggressive GQA
(group = 8, only 4 KV heads) is used to shrink that KV read. This is the long-context
variant of PagedAttention decode where the KV read dominates end-to-end latency.

This Model.forward IS the naive reference: it gathers each sequence's pages into a
contiguous [seq_len, Hkv, D] tensor, expands KV heads to query heads (GQA), and does
a plain softmax(q·kᵀ/√d)·v in fp32. Correct, but slow. It is the CORRECTNESS oracle.

STRONG BASELINE to beat (not naive torch): flashinfer
    BatchDecodeWithPagedKVCacheWrapper  (fused paged-attention CUDA kernel).
Inputs are fp16 to match serving; the naive ref upcasts to fp32.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Naive long-context paged-decode attention. Gather pages + softmax fp32. Slow."""

    def __init__(self):
        super().__init__()

    def forward(self, q, kv_cache, kv_indptr, kv_indices, last_page_len):
        B, H, D = q.shape
        page_size, Hkv = kv_cache.shape[2], kv_cache.shape[3]
        group = H // Hkv
        scale = 1.0 / (D ** 0.5)
        qf = q.float()
        out = torch.empty(B, H, D, dtype=torch.float32, device=q.device)
        for b in range(B):
            p0, p1 = int(kv_indptr[b]), int(kv_indptr[b + 1])
            pages = kv_indices[p0:p1]
            npg = pages.numel()
            lpl = int(last_page_len[b])
            seq_len = (npg - 1) * page_size + lpl
            k = kv_cache[pages, 0].reshape(npg * page_size, Hkv, D)[:seq_len].float()
            v = kv_cache[pages, 1].reshape(npg * page_size, Hkv, D)[:seq_len].float()
            k = k.repeat_interleave(group, dim=1)         # (S, H, D)
            v = v.repeat_interleave(group, dim=1)
            scores = torch.einsum("hd,shd->hs", qf[b], k) * scale
            probs = torch.softmax(scores, dim=-1)
            out[b] = torch.einsum("hs,shd->hd", probs, v)
        return out.to(q.dtype)


# --- workload: (B, num_qo_heads, num_kv_heads, head_dim, seq_len, page_size) ---
# aggressive GQA (group = 8, Hkv = 4), very long contexts 2048..8192.
WORKLOAD = [
    (2, 32, 4, 128, 2048, 16),   # small smoke (still long ctx)
    (4, 32, 4, 128, 4096, 16),   # medium
    (4, 32, 4, 128, 8192, 16),   # long
    (8, 32, 4, 128, 8192, 16),   # large batch, very long ctx
]

_DEF = (2, 32, 4, 128, 2048, 16)


def _build_paged(shape, device, seed):
    B, H, Hkv, D, S, page_size = shape
    torch.manual_seed(seed)
    assert S % page_size == 0, "seq_len must be a multiple of page_size"
    pages_per_seq = S // page_size
    total_pages = B * pages_per_seq
    q = (torch.randn(B, H, D, device=device) * 0.5).half()
    kv_cache = (torch.randn(total_pages, 2, page_size, Hkv, D, device=device) * 0.5).half()
    kv_indptr = (torch.arange(0, B + 1, device=device, dtype=torch.int32) * pages_per_seq)
    kv_indices = torch.arange(0, total_pages, device=device, dtype=torch.int32)
    last_page_len = torch.full((B,), page_size, device=device, dtype=torch.int32)
    return [q, kv_cache, kv_indptr, kv_indices, last_page_len]


def make_inputs(shape, device, seed=0):
    return _build_paged(shape, device, seed)


def get_inputs():
    return _build_paged(_DEF, "cuda", 0)


def get_init_inputs():
    return []


OUTPUT_INDEX = 0

_WS = {}


def baseline(q, kv_cache, kv_indptr, kv_indices, last_page_len, init_args=None):
    """Strong baseline: flashinfer BatchDecodeWithPagedKVCacheWrapper (long ctx GQA)."""
    import flashinfer
    B, H, D = q.shape
    page_size, Hkv = kv_cache.shape[2], kv_cache.shape[3]
    dev = q.device
    if dev not in _WS:
        _WS[dev] = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    wrap = flashinfer.BatchDecodeWithPagedKVCacheWrapper(_WS[dev], "NHD")
    wrap.plan(kv_indptr, kv_indices, last_page_len, H, Hkv, D, page_size,
              q_data_type=q.dtype, kv_data_type=kv_cache.dtype)
    return wrap.run(q, kv_cache)  # (B, H, D)
