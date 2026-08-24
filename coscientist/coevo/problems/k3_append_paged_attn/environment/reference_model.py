"""K3 task: Append / chunked-prefill attention over a paged KV cache.

REAL production hotspot: continuous-batching LLM serving (vLLM) does *append*
attention — each request already has a KV cache of `kv_len` tokens stored in a
paged cache, and a chunk of `q_len` NEW query tokens is appended this step. Each
new query token attends over the whole cache (prefix + the new tokens up to and
including itself), with causal masking anchored at its absolute position
(kv_len - q_len + i). This is the unifying kernel behind both prefill (q_len ==
kv_len) and speculative/chunked decode (q_len << kv_len).

This Model.forward IS the naive reference: it gathers each request's KV pages
into a contiguous [kv_len, Hkv, D] tensor, expands KV heads to query heads (GQA),
and does a plain position-anchored causal softmax(q·kᵀ/√d)·v in fp32, per request.
Correct, but slow (python loop over batch + full materialized scores). It is the
CORRECTNESS oracle.

STRONG BASELINE to beat (not naive torch): flashinfer
    BatchPrefillWithPagedKVCacheWrapper(..., causal=True)
(the fused paged append-attention CUDA kernel). Inputs are fp16 to match serving.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Naive paged append attention. Gathers pages + position-anchored causal
    softmax in fp32, per request. Slow. Shape-agnostic (head dims from tensors)."""

    def __init__(self):
        super().__init__()

    def forward(self, q, qo_indptr, kv_cache, kv_indptr, kv_indices, last_page_len):
        # q         : (total_q, H, D)                       appended query tokens (ragged)
        # qo_indptr : (B+1,) int32                          per-request query ranges
        # kv_cache  : (num_pages, 2, page_size, Hkv, D)     paged K/V (NHD layout)
        # kv_indptr : (B+1,) int32                          page ranges per request
        # kv_indices: (num_used_pages,) int32               page ids
        # last_page_len : (B,) int32                        valid entries in last page
        H, D = q.shape[1], q.shape[2]
        page_size, Hkv = kv_cache.shape[2], kv_cache.shape[3]
        group = H // Hkv
        scale = 1.0 / (D ** 0.5)
        B = qo_indptr.numel() - 1
        out = torch.empty(q.shape[0], H, D, dtype=torch.float32, device=q.device)
        for b in range(B):
            qs, qe = int(qo_indptr[b]), int(qo_indptr[b + 1])
            q_len = qe - qs
            p0, p1 = int(kv_indptr[b]), int(kv_indptr[b + 1])
            pages = kv_indices[p0:p1]
            npg = pages.numel()
            lpl = int(last_page_len[b])
            kv_len = (npg - 1) * page_size + lpl
            k = kv_cache[pages, 0].reshape(npg * page_size, Hkv, D)[:kv_len].float()
            v = kv_cache[pages, 1].reshape(npg * page_size, Hkv, D)[:kv_len].float()
            k = k.repeat_interleave(group, dim=1)          # (kv_len, H, D)
            v = v.repeat_interleave(group, dim=1)
            qb = q[qs:qe].float()                          # (q_len, H, D)
            scores = torch.einsum("qhd,khd->hqk", qb, k) * scale  # (H, q_len, kv_len)
            # query i has absolute position (kv_len - q_len + i); attends keys j<=pos
            qpos = torch.arange(kv_len - q_len, kv_len, device=q.device)  # (q_len,)
            kpos = torch.arange(kv_len, device=q.device)                  # (kv_len,)
            mask = kpos[None, :] > qpos[:, None]           # (q_len, kv_len) masked-out
            scores = scores.masked_fill(mask[None], float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out[qs:qe] = torch.einsum("hqk,khd->qhd", probs, v)  # (q_len, H, D)
        return out.to(q.dtype)


# --- workload: (per_req_(q_len,kv_len) tuples, num_qo_heads, num_kv_heads, head_dim, page_size)
# GQA H=Hkv*group, head_dim 128, 16-token pages. kv_len >= q_len per request.
WORKLOAD = [
    (((16, 256), (32, 512)),                              32, 8, 128, 16),  # small smoke
    (((32, 512), (64, 1024), (16, 256)),                  32, 8, 128, 16),  # medium
    (((64, 1024), (128, 2048), (32, 512), (16, 256)),     32, 8, 128, 16),  # long
    (((256, 2048), (128, 4096), (64, 1024)),              40, 8, 128, 16),  # very long, big q chunks
]

_DEF = (((16, 256), (32, 512)), 32, 8, 128, 16)


def _build(shape, device, seed):
    reqs, H, Hkv, D, page_size = shape
    torch.manual_seed(seed)
    qo_indptr = [0]
    kv_indptr = [0]
    last_page_len = []
    total_q = 0
    total_pages = 0
    for (q_len, kv_len) in reqs:
        assert kv_len >= q_len
        total_q += q_len
        qo_indptr.append(total_q)
        npg = (kv_len + page_size - 1) // page_size
        lpl = kv_len - (npg - 1) * page_size
        total_pages += npg
        kv_indptr.append(total_pages)
        last_page_len.append(lpl)
    q = (torch.randn(total_q, H, D, device=device) * 0.5).half()
    kv_cache = (torch.randn(total_pages, 2, page_size, Hkv, D, device=device) * 0.5).half()
    qo_indptr = torch.tensor(qo_indptr, device=device, dtype=torch.int32)
    kv_indptr = torch.tensor(kv_indptr, device=device, dtype=torch.int32)
    kv_indices = torch.arange(total_pages, device=device, dtype=torch.int32)
    last_page_len = torch.tensor(last_page_len, device=device, dtype=torch.int32)
    return [q, qo_indptr, kv_cache, kv_indptr, kv_indices, last_page_len]


def make_inputs(shape, device, seed=0):
    return _build(shape, device, seed)


def get_inputs():
    return _build(_DEF, "cuda", 0)


def get_init_inputs():
    return []  # shape-agnostic


OUTPUT_INDEX = 0

_WS = {}


def baseline(q, qo_indptr, kv_cache, kv_indptr, kv_indices, last_page_len, init_args=None):
    """Strong baseline: flashinfer BatchPrefillWithPagedKVCacheWrapper (append, causal)."""
    import flashinfer
    H, D = q.shape[1], q.shape[2]
    page_size, Hkv = kv_cache.shape[2], kv_cache.shape[3]
    dev = q.device
    if dev not in _WS:
        _WS[dev] = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    wrap = flashinfer.BatchPrefillWithPagedKVCacheWrapper(_WS[dev], "NHD")
    wrap.plan(qo_indptr, kv_indptr, kv_indices, last_page_len, H, Hkv, D, page_size,
              causal=True, q_data_type=q.dtype, kv_data_type=kv_cache.dtype)
    return wrap.run(q, kv_cache)  # (total_q, H, D)
