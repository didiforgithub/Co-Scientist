# AES-128-CTR encryption throughput

Optimize the C function `aes128_ctr_encrypt()` so that encrypting a 256 MiB
buffer in AES-128 counter mode runs as fast as possible on a single CPU core,
while still producing bit-exact correct output.

A candidate solution is the **full source of `solve.c`** implementing the
interface fixed in `solve.h`:

```c
void aes128_ctr_encrypt(const uint8_t key[16], const uint8_t iv[16],
                        const uint8_t *plaintext, uint8_t *ciphertext,
                        size_t len);
```

CTR mode is self-inverse and must match NIST SP 800-38A Appendix F.5.1 exactly.

## What ships in `environment/`

| File | Role |
|------|------|
| `solve.c` | **Seed / baseline** candidate: a portable scalar byte-level AES with S-box + `xtime` MixColumns, sequential CTR blocks. Correct but slow (~3 s). This is the starting candidate a solver improves. |
| `solve.h` | Fixed interface. The signature must not change. |
| `main.c` | **Timed benchmark harness.** Fills a 256 MiB deterministic buffer, runs one warm-up call then 5 timed calls, prints `runs=.. time=<median_seconds> result=ok`. Gates on the NIST F.5.1 vector before timing. Read-only in the task. |
| `Makefile` | Fixed build: `gcc -O2 -march=native -std=c99`. Do not modify. |
| `Dockerfile` | The scoring environment: `ubuntu:22.04` + gcc/make/python3, single core budget. |
| `verify_correctness.py` | **Correctness oracle.** Compiles `solve.c` into a small harness and checks its AES-128-CTR output against a pure-Python reference on the NIST vector plus a battery of random keys/IVs/lengths (empty, sub-block, block-aligned, unaligned, up to a few hundred bytes). Exit 0 = all pass. |

## How a candidate should be judged

The metric is **speedup over the baseline**: `speedup = baseline_seconds /
candidate_median_seconds`, mapped to a bounded reward (a log-scaled clip; the
reference AES-NI-class solution reaches ~0.10 s ≈ 30x). A candidate must first
pass the correctness oracle on the NIST vector **and** the randomized cases —
correctness is a hard gate, wrong answers score zero — and only then is its
benchmark time turned into a reward. The optimization is single-threaded; the
candidate may use standard C, `<immintrin.h>` / `<wmmintrin.h>` intrinsics, and
GCC extensions, but no external libraries, no threads, and no network.

The correct throughput must be a property of the AES computation itself on
*arbitrary* inputs — not of any one fixed benchmark buffer, key, IV, or of the
harness calling convention. A candidate that only appears fast because it
recognizes the specific benchmark workload, reuses results across repeated
timed calls, or influences the measured time is not actually solving the
problem and should not score well.

## Baseline / reference anchors

- Baseline (the shipped scalar `solve.c`): ~3.0 s median.
- Reference (AES-NI hardware intrinsics + 8-way CTR parallelism): ~0.10 s median.
