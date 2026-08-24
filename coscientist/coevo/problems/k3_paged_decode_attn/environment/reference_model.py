"""K3 task: Paged KV-cache decode attention (the LLM inference decode hotspot).

REAL production hotspot: during autoregressive decoding, every new token attends
over the entire cached K/V history, which is stored in a *paged* KV cache (vLLM /
PagedAttention style) — physically non-contiguous 16-token pages indexed by a
per-sequence page table. This single-query-token attention over long paged history
is the dominant cost of LLM serving.

This Model.forward IS the naive reference: it gathers each sequence's pages back
into a contiguous [seq_len, H_kv, D] tensor, expands KV heads to query heads (GQA),
and does a plain softmax(q·kᵀ/√d)·v in fp32. Correct, but slow (python loop over
batch + full materialized scores). It is the CORRECTNESS oracle.

STRONG BASELINE to beat (not naive torch): flashinfer
    BatchDecodeWithPagedKVCacheWrapper  (fused paged-attention CUDA kernel).
Inputs are fp16 to match the serving-time kernel; the naive ref upcasts to fp32.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Naive paged-decode attention. Gathers pages + full softmax in fp32. Slow.

    Shape-agnostic: head counts / page size are read from the input tensors, so a
    single Model instance handles every WORKLOAD shape (get_init_inputs is empty).
    """

    def __init__(self):
        super().__init__()

    def forward(self, q, kv_cache, kv_indptr, kv_indices, last_page_len):
        # q         : (B, num_qo_heads, head_dim)   single decode token per seq
        # kv_cache  : (num_pages, 2, page_size, num_kv_heads, head_dim)  NHD layout
        # kv_indptr : (B+1,)  int32  page ranges per seq
        # kv_indices: (num_used_pages,) int32  page ids
        # last_page_len: (B,) int32  valid entries in each seq's last page
        B, H, D = q.shape
        page_size, Hkv = kv_cache.shape[2], kv_cache.shape[3]
        group = H // Hkv
        scale = 1.0 / (D ** 0.5)
        qf = q.float()
        out = torch.empty(B, H, D, dtype=torch.float32, device=q.device)
        for b in range(B):
            p0, p1 = int(kv_indptr[b]), int(kv_indptr[b + 1])
            pages = kv_indices[p0:p1]                     # page ids for this seq
            npg = pages.numel()
            lpl = int(last_page_len[b])
            k_pg = kv_cache[pages, 0]                      # (npg, page_size, Hkv, D)
            v_pg = kv_cache[pages, 1]
            k = k_pg.reshape(npg * page_size, Hkv, D).float()
            v = v_pg.reshape(npg * page_size, Hkv, D).float()
            seq_len = (npg - 1) * page_size + lpl          # drop padding on last page
            k = k[:seq_len]                               # (S, Hkv, D)
            v = v[:seq_len]
            # expand kv heads to query heads (GQA)
            k = k.repeat_interleave(group, dim=1)         # (S, H, D)
            v = v.repeat_interleave(group, dim=1)
            qb = qf[b]                                     # (H, D)
            # scores: (H, S)
            scores = torch.einsum("hd,shd->hs", qb, k) * scale
            probs = torch.softmax(scores, dim=-1)
            out[b] = torch.einsum("hs,shd->hd", probs, v)  # (H, D)
        return out.to(q.dtype)


# --- workload: (B, num_qo_heads, num_kv_heads, head_dim, seq_len, page_size) ---
# GQA (H = Hkv*group), head_dim 128, 16-token pages. seq_len multiple of page_size.
WORKLOAD = [
    (4,  32, 8, 128, 256,  16),   # small smoke
    (8,  32, 8, 128, 512,  16),   # medium
    (16, 32, 8, 128, 1024, 16),   # long context
    (32, 64, 8, 128, 2048, 16),   # large batch, long context, group=8 (Llama3-70B GQA)
]

_DEF = (4, 32, 8, 128, 256, 16)


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
    return []  # Model is shape-agnostic; head dims come from the input tensors


OUTPUT_INDEX = 0


# reuse one wrapper (workspace) per process
_WS = {}


def baseline(q, kv_cache, kv_indptr, kv_indices, last_page_len, init_args=None):
    """Strong baseline: flashinfer BatchDecodeWithPagedKVCacheWrapper."""
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
