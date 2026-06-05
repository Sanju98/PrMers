# PrMers — Optimization & Parallelism Opportunities

A code-level audit of PrMers for performance and parallelism opportunities,
organized by **implementation effort**. Findings come from reading the source
(host orchestration, OpenCL kernels, math layer, and app-level concurrency).

## Context

- **Hardware audited:** Apple M4 Pro (14 CPU cores, 20-core GPU, OpenCL 1.2, no FP64).
- **Active engine on Apple:** the **Marin** engine (`include/marin/`, `src/modes/RunPrpOrLlMarin.cpp`, `RunLlSafeMarin.cpp`). The older `NttEngine`/`kernels/prmers.cl` path ("Legacy") runs on other setups.
- **Baseline:** ~1,059 iterations/sec for exponent 6,972,593 (FFT 327,680 words = 5·2¹⁶), ~5 GPU kernels per squaring.
- **Status:** Originally **static-analysis findings** (read, not benchmarked). A subset has since been implemented and measured — see [Measured results](#measured-results-apple-m4-pro) at the end. Validate any remaining item with an A/B IPS run before trusting its impact estimate.

## Legend

- **Impact:** High / Med / Low (estimated effect on the metric noted).
- **Engine:** Marin (runs on Apple) · Legacy · Both · ECM.
- **⊕ convergent** = independently flagged by 2+ audit passes (higher confidence).
- The single biggest theme: **PrMers treats the Apple GPU as a generic "non-NVIDIA" device** — it is lumped into a conservative path with no Apple/unified-memory tuning.

---

## 🟢 Low effort (tweaks, host-side constants, hoisting)

### L1 ⊕ — Apple sync cadence forces periodic full GPU drains  · Impact: Med–High · Marin
- **Where:** `include/marin/ocl.h:297` (`if (_vendor != NVIDIA) _is_sync = true;`), `:574-578` (`clFinish` every 16384 enqueues), vendor detect `:319-331`.
- **Why:** At ~5 kernels/iter, a blocking `clFinish` lands every ~3,200 iterations. Apple's deprecated OpenCL-over-Metal submission is comparatively expensive, so a global drain that often serializes CPU enqueue with GPU idle. Apple reports vendor "Apple" → falls into the generic non-NVIDIA branch with no tuning.
- **Change:** Detect Apple explicitly; widen the interval (e.g. 64K–256K) or use `clFlush` instead of `clFinish`.
- **Risk:** Low — enqueues are already ordered in one in-order queue; correctness unaffected.
- **Recommended first experiment** (one-line change, directly measurable via IPS).

### L2 — Branchless `reduce()` tail on Apple  · Impact: High · Marin
- **Where:** `kernels/marin.cl:64-91` (`reduce`), gated by `PTX_ASM` at `:14-16,72`.
- **Why:** The fast carry-flag reduction is NVIDIA-only (`PTX_ASM`); Apple takes the generic branch with two data-dependent conditionals on the hottest path (every butterfly + pointwise square).
- **Change:** Use a branchless mask form (`r -= MOD_P & -(r>=s)`) so Apple gets predication-free straight-line code. Drop-in.
- **Risk:** Low — algebraically identical; guarded by existing Gerbicz check.

### L3 — Hoist the ECM engine out of the per-curve `-K` loop  · Impact: High (ECM) · ECM
- **Where:** `src/modes/RunEcm.cpp:672` (curve loop) `:690` (`engine::create_gpu` per curve) + per-curve `delete`.
- **Why:** Every curve rebuilds the whole engine: OpenCL program reload + 51-register alloc + roots/weights/widths upload (`engine_gpu.h:648-705`). Pure repeated startup cost × K.
- **Change:** Build the engine once before the loop, re-init only per-curve state.
- **Risk:** Low — register file already has room; curves are independent.

### L4 — Apple work-group / local-memory tuning (host constants)  · Impact: Med · Marin
- **Where:** `include/marin/engine_gpu.h:78-93` (CHUNK/BLK formulas), `marin.cl` `reqd_work_group_size` attrs (`:1038,1482,1599`).
- **Why:** CHUNK/BLK are capped for AMD's old ≤4 KB local-memory budget; the 320-stage uses only ~10 KB of Apple's 32 KB. Raising CHUNK (e.g. 320-stage CHUNK 2→3) does more work per dispatch → fewer launches.
- **Change:** Add an Apple branch that exploits the full 32 KB local; benchmark WG 64/128/160/256 for `sqr1024`/carry instead of pinning 256.
- **Risk:** Low — kernels are already macro-parameterized on CHUNK/BLK.

### L5 — Legacy: hoist `clSetKernelArg` out of the per-iteration loop  · Impact: Med · Legacy
- **Where:** `src/opencl/NttEngine.cpp:269-374` → `include/opencl/NttPipeline.hpp:53-79`.
- **Why:** ~40–60 redundant `clSetKernelArg` host calls/iteration; args are loop-invariant (only `buf_x` changes, and it's constant within a run). Marin already binds args once.
- **Change:** Set args once after pipeline build; only enqueue in the loop.
- **Risk:** Low (does not affect the Apple/Marin path).

### L6 — Legacy: drop `CL_QUEUE_PROFILING_ENABLE` on the hot queue  · Impact: Low–Med · Legacy
- **Where:** `src/opencl/Context.cpp:192`.
- **Why:** Profiling timestamps every command even though the legacy hot loop never reads them. Marin already keeps a separate non-profiling fast queue.
- **Change:** Enable profiling only when tuning/profiling is requested.

### L7 — Legacy: unify transform-size on the tight bound + delete magic override  · Impact: High (affected band) · Legacy
- **Where:** `src/math/Precompute.cpp:38` (`>=63`) vs `include/marin/ibdwt.h:28` (`>=64`); magic `if(exponent>1207959503)` at `Precompute.cpp:51-53`.
- **Why:** The legacy picker leaves an extra bit of headroom (`>=63`), selecting the next-larger N one exponent-band early → up to ~1.25–2× more words/iteration for affected exponents, no correctness benefit. Marin's `>=64` is the provably-tight bound.
- **Change:** Adopt Marin's `transform_size` logic for the legacy path; remove the magic override. (Marin/Apple path already uses the tight bound.)
- **Risk:** Med — re-validate Gerbicz at band boundaries.

---

## 🟡 Medium effort (new kernels, buffer repacking, threading, fusion)

### M1 ⊕ — GPU-side Gerbicz compare instead of host residue round-trip  · Impact: Med–High · Marin
- **Where:** `src/modes/RunPrpOrLlMarin.cpp:331-337`, `RunLlSafeMarin.cpp:273-281,474-483` (`get_mpz` → blocking read + GMP `% Mp`). LL-safe does it 4×.
- **Why:** Each check drains the queue (`clFinish`) and DMAs full residues to host, then big-int compares on CPU while the GPU idles. The Legacy path already does this on-GPU and reads back one `cl_uint` (`RunPrpOrLl.cpp:628-637`).
- **Change:** Add a Marin `check_equal` register kernel; read back a single OK flag. Keep `get_mpz` only for final result / res64 / checkpoint.
- **Risk:** Med — residues may differ by a multiple of Mp; match canonical-form semantics carefully.

### M2 ⊕ — Background, double-buffered checkpointing  · Impact: High · Both
- **Where:** `RunPrpOrLlMarin.cpp:281-288` → `engine_gpu.h:1348-1352` → `ocl.h:485,514-521`. Legacy is worse: 5 blocking reads + 4 writes per backup (`RunPrpOrLl.cpp:687-732`).
- **Why:** Every backup interval the loop thread does `clFinish` + blocking read of the **entire 8N-word register file** + CRC32 + disk write — GPU fully idle for the whole duration (tens–hundreds of MB at large N).
- **Change:** On the loop thread do only the device→host copy into a swappable pre-allocated buffer; hand off CRC + file write + atomic rename to a background writer thread (the codebase already uses detached threads for res64 display). Ensure one in-flight writer + interrupt-path flush.
- **Risk:** Med.

### M3 — Fuse carry-pass-1 into the backward-FFT store  · Impact: High · Marin
- **Where:** `engine_gpu.h:826,839`; `marin.cl:965-977,1053,1600`.
- **Why:** Each squaring makes ~5 full passes over the 2.6 MB word array. `backward320_0` stores to global only for `carry_weight_mul_p1` to immediately re-read it; the last FFT stage already holds the word in registers.
- **Change:** Fold the weight/unweight + carry-pass-1 into the backward store, removing one full read+write (~15–20%). Keep pass-2 separate (inter-group carry).
- **Risk:** Med — carry correctness across work-groups is the reason for the two-pass design.

### M4 — Verify / hand-code the 64×64→128 `mul_hi` emulation  · Impact: High · Marin
- **Where:** `marin.cl:95` (`mod_mul` uses `mul_hi(ulong,ulong)`).
- **Why:** Apple GPUs have **no native 64-bit high-multiply**; the compiler emulates it (several 32-bit ops) — the single largest per-butterfly ALU cost. This is partly a hardware floor for integer NTT on Apple.
- **Change:** Confirm the compiler isn't emitting a full software 128-bit routine; if it is, inline a tight 4×`mul_hi(uint)`/`mad_hi` sequence. (Longer term: evaluate the GF(2³¹−1) path to halve multiply width.)
- **Risk:** Med — validate against existing residue/Gerbicz checks.

### M5 — Repack IBDWT weights for contiguous vector loads  · Impact: Med · Marin
- **Where:** `marin.cl:1610…` (`loadg2(...,N/4)` strided gather), layout in `ibdwt.h:137`.
- **Why:** Carry kernels read 4 weights with stride N/4 (4 cache lines/thread, ~640 KB apart → no L2 reuse), twice per iteration.
- **Change:** Pre-pack weights as interleaved `uint64_4` in digit-major order so the carry kernel does one contiguous vector load.
- **Risk:** Med — repack weight + width buffers and update all ~12 carry kernels consistently.

### M6 — Exploit unified memory for residue reads  · Impact: Med · Marin (Apple)
- **Where:** `ocl.h:489-528` (blocking `CL_TRUE` reads, plain `CL_MEM_READ_WRITE`), `engine_gpu.h:122-126`.
- **Why:** Host and device share physical RAM on M-series, but PrMers does full memcpy + barrier reads as if discrete.
- **Change:** Allocate staging buffers with `CL_MEM_ALLOC_HOST_PTR` and use `clEnqueueMapBuffer`/`Unmap` (near-zero-copy) for checkpoint/Gerbicz reads. Pairs with M1/M2.
- **Risk:** Med — Apple-specific branch; verify map/unmap vs in-flight queue.

### M7 — Parallel-prefix carry pass-2 (replace serial ripple)  · Impact: Med · Legacy/Both
- **Where:** `kernels/prmers.cl:542-549` (`kernel_carry_2` reads `carry_array[gid-1]`).
- **Why:** Pass-2 is an O(workers) serial inter-block ripple (works only because carries usually die within a block via early-exit). Worst case is a serial chain + a global barrier between the two carry launches.
- **Change:** Replace the linear ripple with a Hillis-Steele / block-scan parallel prefix (O(log workers)), or fuse into a single kernel with a local scan + one atomic spill.
- **Risk:** Med — carry correctness is the whole test; lean on Gerbicz as the net.

### M8 — Remove redundant re-square inside the Gerbicz check branch  · Impact: Med · Legacy
- **Where:** `src/modes/RunPrpOrLl.cpp:588-617` (extra `forward(input)/inverse(input)` at `:602-603,616-617`).
- **Why:** The check branch appears to recompute a squaring already done in the main loop. Amortized over block size B, but B is small early on.
- **Change:** Cache and reuse the main-loop square; confirm whether both transforms are genuinely independent recomputations before removing.
- **Risk:** Med — Gerbicz correctness is subtle.

---

## 🔴 High effort (architecture, new backends)

### H1 ⊕ — Multi-GPU / multi-exponent concurrency  · Impact: High (throughput) · Both
- **Where:** `RunPrpOrLlMarin.cpp:683-718` / `RunPrpOrLl.cpp:1015-1062` (serial worktodo via `restart_self`); `device_id` is a single scalar (`engine_gpu.h:648`).
- **Why:** Exponents run strictly one-at-a-time; no per-device fan-out. Independent exponents are embarrassingly parallel.
- **Change:** Worker-per-device model — parse worktodo into a shared queue, spawn one App/engine per `device_id`. Replaces the `restart_self` lifecycle with a long-lived dispatcher.
- **Note:** Throughput, not single-test latency (one large exponent already fills one GPU).
- **Risk:** High — touches engine lifetime, buffer ownership, worktodo lifecycle.

### H2 — Batch independent ECM curves over shared FFT kernels  · Impact: High (ECM) · ECM
- **Where:** `RunEcm.cpp:672` (after the L3 hoist).
- **Why:** A single Mersenne FFT under-utilizes a 20-core GPU at small/medium N; independent curves are parallel and could vectorize over the register file or run as multi-queue workers.
- **Risk:** High — curve-batched FFT layout is a substantial kernel change.

### H3 — Metal backend (the real Apple unlock)  · Impact: High (Apple) · new
- **Why:** PrMers runs Apple's **deprecated** OpenCL 1.2 shim over Metal. A native Metal compute backend would remove the legacy-path penalties (L1, M6) at the source and unlock Apple-specific scheduling/threadgroup-memory tuning.
- **Risk:** Very high — a new backend, not a tweak. Listed for completeness; the Tier-1/2 Apple tweaks capture most of the gain at a fraction of the cost.

---

## Suggested order

1. **L1** (Apple sync cadence) — one line, measurable, lowest risk. **Do first.**
2. **L2 + L4** (branchless reduce + Apple local-mem/WG tuning) — cheap kernel/host wins on the hot path.
3. **M2 + M1** (background checkpoint + GPU-side Gerbicz compare) — remove the only hard pipeline drains on the Marin path.
4. **M3 + M6** (carry fusion + unified-memory maps) — attack the memory-bound side.
5. **M4** (mul_hi) — investigate the dominant ALU cost / Apple hardware floor.
6. **H1 / H2** — when multi-GPU throughput or ECM matters.
7. **H3** — only if Apple becomes a primary target and the tweaks above prove insufficient.

> Each item should be A/B-benchmarked with `./prmers 6972593 -ll` (or a fixed exponent) against the ~1,059 IPS baseline before and after.

---

## Measured results (Apple M4 Pro)

Serial benchmarks (one GPU run at a time). **Baseline** = unmodified upstream
PrMers; **Current** = this branch (M6 unified-memory reads + ECM engine hoist +
the runtime-capability / RAII cleanup).

### Latency score

| workload | baseline | current | delta |
|---|---|---|---|
| **Lucas-Lehmer** `M216091 -ll` | 33.55 s  (≈6,440 IPS) | 33.49 s  (≈6,410 IPS) | **≈ 0% (no regression)** |
| **ECM** `M9941 -b1 2000 -K 8` | 33.3 s | **24.1 s** | **−27%** |

Both verdicts correct (LL → prime; ECM → identical work, no factor). The ECM
win is the engine hoist removing K−1 per-curve engine rebuilds; LL is untouched
on the hot path, so it holds at baseline.

### Per-experiment verdicts (hypotheses → tested facts)

| change | hypothesis | measured | kept? |
|---|---|---|---|
| **L1** Apple sync cadence | top win | **−24% (regression)** — Apple prefers frequent drains; hypothesis was backwards | ❌ rejected |
| **L2** branchless reduce | faster butterfly | neutral | ❌ dropped (no benefit) |
| **M6** unified-memory reads | fewer stalls | LL-neutral; shrinks checkpoint/Gerbicz stalls (not exercised by short runs) | ✅ kept |
| **ECM hoist** (L3) | less per-curve overhead | **−27% ECM** | ✅ kept |

Lesson: the audit's #1 pick (L1) was a regression on real silicon, while the
unglamorous ECM-hoist + unified-memory changes are the keepers — **measure
before merging.**
