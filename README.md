# Open Kernel Experiments

A stopped, Codex-assisted GPU optimization experiment for the [Muse Spark Open Kernel Challenge](https://openkernelchallenge.dev/), tested locally on Modal H200 GPUs and against the official evaluator.

**We improved on the SGLang baselines, but did not beat the leaderboard leaders.** This repository preserves the useful kernels, slower alternatives, correctness failures, measurements, and unfinished work. It is an experiment archive, not a production kernel library.

Snapshot: **2026-10-24**. Optimization was stopped at the owner's request. All Modal apps were stopped with zero running tasks. Previously queued official challenge jobs may finish later; this archive does not automatically track them.

## Official scored results

These are completed **scored submissions**, not local estimates or off-board checks. All five passed the evaluator with no reported flags. Speedup is relative to the challenge's SGLang baseline; a larger number is better. These five speedups must not be averaged to infer an overall challenge score.

| Problem | Speedup | Reported 95% interval | Exact source | Official job |
|---|---:|---:|---|---|
| Attention decode | **1.4807×** | 1.4584–1.5086× | [attention_fused_hybrid.py](outputs/attention_fused_hybrid.py) | `b4833dc1766c4451` |
| BF16 projection | **1.2195×** | 1.2012–1.2377× | [dense_warp_hybrid.py](outputs/dense_warp_hybrid.py) | `8eb008273d6c4ec6` |
| Whole decoder layer | **1.1852×** | 1.1674–1.2030× | [decode_fused_boundaries.py](outputs/decode_fused_boundaries.py) | `59fa310d73a342ea` |
| Routed NVFP4 experts | **1.0971×** | 1.0806–1.1135× | [moe_candidate.py](outputs/moe_candidate.py) | `1ca1569ded2347ef` |
| Prefill layer | **1.1206×** | 1.1038–1.1374× | [prefill_grouped_tma.py](outputs/prefill_grouped_tma.py) | `762a6d1810654813` |

[Machine-readable manifest](outputs/best_manifest.json) includes exact source SHA-256 hashes, server content identifiers, earlier scored results and receipt paths. Official JSON files are selected-field excerpts; host paths, logs and machine diagnostics were omitted. Challenge positions remain provisional until organizer review.

### Later checks, awaiting scored results

| Candidate | Official off-board check | Scored job at snapshot |
|---|---:|---|
| [Balanced attention](outputs/attention_balanced_hybrid.py) | 1.4939× | `509471d6c70c4e0f`, queued |
| [Compact decoder routing](outputs/decode_compact_align.py) | 1.1989× | `8290bde5fbb24dbd`, queued |
| [CUDA MoE routing alignment](outputs/moe_cuda_align.py) | 1.1550× | `d807437847834796`, queued |

These checks are **not leaderboard scores**. Small changes often have overlapping intervals. The [official job snapshot](results/official-jobs-snapshot.json) also records other pending checks and submissions. No new submissions were made after the stop request.

## What worked, and what did not

- **Useful:** custom attention kernels with shape-based fallback; small-batch dense projection kernels; fusing decoder boundaries; grouped TMA prefill; compact expert routing and CUDA histogram/prefix/scatter alignment.
- **Small local gains only:** compact expert dispatch reached 1.1715× and 1.1696× in two local runs, versus 1.1595× and 1.1598× controls. Its decoder integration reached 1.2506× versus 1.2414×, with 50/50 recorded local checks. Official checks were still queued. See [compact dispatch](results/research/compact-dispatch-research.json).
- **Rejected:** custom Marlin stripe scheduling passed 76/76 checks but achieved about 1.066–1.077× versus a 1.167× control. See [stripe experiments](results/research/aligned-stripes-research.json).
- **Rejected:** exact register-byte FP4 lookup passed its bit-level proof and 76/76 benchmark checks, but its GEMV variants were slower than the native implementation. See [PRMT experiments](results/research/prmt-gemv-research.json).
- **Rejected:** two-stage attention, adaptive expert tiles, scaled-bit FP4 conversions, asynchronous weight prefetch and many other tiling/kernel variants. Correctness alone did not imply speed.
- **Unfinished:** a group-16 E4M3 NVFP4 Hopper mixed-input GEMM adaptation passed a 16,384-value scale-layout probe and 10 small GEMM checks, including fresh-input CUDA graph replays. Maximum observed relative L2 error was about 0.00173. It was **not integrated into full MoE, benchmarked for speed, or submitted officially**. See [prototype notes](docs/hopper-prototype.md).

The archive contains **115 local benchmark records** and **627 distinct candidate source snapshots**. The records contain 18,053 passing checks out of 18,121 recorded checks, including repeated controls and unsuccessful experiments. These are not 18,121 independent tests, nor a pass rate for the final submissions. Some early records lack source snapshots; the [index](results/local/index.json) identifies them.

## Measurement method

The local harness used an H200 and pinned image:

```text
lmsysorg/sglang@sha256:8cfc5a72cb296771f5f8a4a4688cb6d05685610a830cbb616fa4ab0eb7bb2b0a
```

Later runs recorded PyTorch 2.14.1+cu130, Triton 3.8.0, Python 3.12.3 and CUDA 13.0. Each record carries its environment and timing method where available. The harness evolved during the experiment; older runs should be interpreted using their own metadata.

The final harness checks correctness before timing, alternates baseline/candidate CUDA graph replays, reads a 128 MiB buffer before the timed interval, and subtracts the measured empty-graph event interval. The default is 30 repetitions; individual experiments used other counts. Failed candidates are excluded from timing. The reference and baseline come from the official challenge files.

Additional MoE routing tests include empty, single-local and all-local routing. The official reference declines empty local routing, so that extra case uses a separate exact-zero BF16 assertion. It does not change official scored cases. [rank_run.py](work/rank_run.py) computes the local load-weighted summary from stored samples. Local estimates and official runs are separate measurements and are not interchangeable.

## Read or reproduce

Reading, hash verification and analysis require no GPU:

```bash
python3 work/verify_archive.py
python3 work/rank_run.py results/local/moe-compact-dispatch.json
```

The exact official candidate files can be checked using the challenge CLI after completing its human login/rules-acceptance flow:

```bash
okc check outputs/dense_warp_hybrid.py --json
okc status 8eb008273d6c4ec6 --json
```

To explicitly run a **new, billable** local replay, see [reproduction instructions](docs/reproduction.md). Nothing here launches automatically. Replays need an authenticated Modal account, the pinned image and official problem files. Current judge files may differ from those used historically; the official receipts retain evaluator hashes. Matching software does not eliminate GPU noise or reproduce a leaderboard position.

## Budget and archive scope

The experiment's compute cap was **$80 including credits**. The final working ledger was **$40.30**, using conservative full-lifecycle GPU estimates plus retained setup/cache/CPU reserves. This is **not an invoice-verified total**. All listed Modal apps were stopped; the persistent compilation cache was retained. See [budget summary](results/budget-summary.json).

- `outputs/`: exact selected submission files, official result excerpts and manifest.
- `results/local/`: timing samples, correctness outcomes and references to candidate snapshots.
- `experiments/sources/`: deduplicated source snapshots named by SHA-256. Includes failed, slower and superseded candidates; do not assume every file is a valid submission.
- `results/research/`: historical experiment notes. Some referenced construction scripts and upstream inspection dumps are not included.
- `results/probes/` and `work/nvfp4_hopper*`: unfinished Hopper proof and evidence.
- `work/`: final local harness, offline analyzer and explicit replay wrapper.

Credentials, account configuration, installed environments, compiler caches, raw logs and downloaded upstream source trees are excluded. Original source notices are preserved; see [third-party provenance](THIRD_PARTY_NOTICES.md).
