"""K3 task: Paged KV-cache decode attention with an FP8 (e4m3) KV cache.

REAL production hotspot: to halve KV-cache memory bandwidth and capacity during
long-context LLM serving, the paged KV cache is stored in FP8 (e4m3). The decode
kernel reads the fp8 pages, dequantizes on the fly, and does single-query-token
attention over the whole cached history (GQA). Query stays in fp16. This is the
fp8-KV variant of the PagedAttention decode hotspot (vLLM / TRT-LLM fp8 KV).

This Model.forward IS the naive reference: it gathers each sequence's fp8 pages,
upcasts (dequantizes) them to fp32, expands KV heads to query heads (GQA), and does
a plain softmax(q·kᵀ/√d)·v in fp32. Correct, but slow. It is the CORRECTNESS oracle.

STRONG BASELINE to beat (not naive torch): flashinfer
    BatchDecodeWithPagedKVCacheWrapper  with kv_data_type=torch.float8_e4m3fn.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Naive fp8-paged decode. Dequantizes pages to fp32 + full softmax. Slow."""

    def __init__(self):
        super().__init__()

    def forward(self, q, kv_cache, kv_indptr, kv_indices, last_page_len):
        # q         : (B, H, D) fp16                          single decode token per seq
        # kv_cache  : (num_pages, 2, page_size, Hkv, D) fp8   NHD paged cache
        # kv_indptr : (B+1,) int32 ; kv_indices : (nnz,) int32 ; last_page_len : (B,) int32
        B, H, D = q.shape
        page_size, Hkv = kv_cache.shape[2], kv_cache.shape[3]
        group = H // Hkv
        scale = 1.0 / (D ** 0.5)
        qf = q.float()
        out = torch.empty(B, H, D, dtype=torch.float32, device=q.device)
        kv_dq = kv_cache.float()  # dequantize fp8 -> fp32
        for b in range(B):
            p0, p1 = int(kv_indptr[b]), int(kv_indptr[b + 1])
            pages = kv_indices[p0:p1]
            npg = pages.numel()
            lpl = int(last_page_len[b])
            seq_len = (npg - 1) * page_size + lpl
            k = kv_dq[pages, 0].reshape(npg * page_size, Hkv, D)[:seq_len]
            v = kv_dq[pages, 1].reshape(npg * page_size, Hkv, D)[:seq_len]
            k = k.repeat_interleave(group, dim=1)         # (S, H, D)
            v = v.repeat_interleave(group, dim=1)
            scores = torch.einsum("hd,shd->hs", qf[b], k) * scale
            probs = torch.softmax(scores, dim=-1)
            out[b] = torch.einsum("hs,shd->hd", probs, v)
        return out.to(q.dtype)


# --- workload: (B, num_qo_heads, num_kv_heads, head_dim, seq_len, page_size) ---
WORKLOAD = [
    (4,  32, 8, 128, 256,  16),   # small smoke
    (8,  32, 8, 128, 512,  16),   # medium
    (16, 32, 8, 128, 1024, 16),   # long context
    (32, 64, 8, 128, 2048, 16),   # large batch, long context, group=8
]

_DEF = (4, 32, 8, 128, 256, 16)


def _build_paged(shape, device, seed):
    B, H, Hkv, D, S, page_size = shape
    torch.manual_seed(seed)
    assert S % page_size == 0, "seq_len must be a multiple of page_size"
    pages_per_seq = S // page_size
    total_pages = B * pages_per_seq
    q = (torch.randn(B, H, D, device=device) * 0.5).half()
    # smaller magnitude so fp8 e4m3 quantization keeps values in-range
    kv_cache = (torch.randn(total_pages, 2, page_size, Hkv, D, device=device) * 0.3).to(torch.float8_e4m3fn)
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
    """Strong baseline: flashinfer BatchDecodeWithPagedKVCacheWrapper (fp8 KV)."""
    import flashinfer
    B, H, D = q.shape
    page_size, Hkv = kv_cache.shape[2], kv_cache.shape[3]
    dev = q.device
    if dev not in _WS:
        _WS[dev] = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    wrap = flashinfer.BatchDecodeWithPagedKVCacheWrapper(_WS[dev], "NHD")
    wrap.plan(kv_indptr, kv_indices, last_page_len, H, Hkv, D, page_size,
              q_data_type=q.dtype, kv_data_type=torch.float8_e4m3fn)
    return wrap.run(q, kv_cache)  # (B, H, D)
