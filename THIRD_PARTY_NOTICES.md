# Provenance and third-party notices

This archive contains original experiment changes and code derived from the Open Kernel Challenge starters/baselines and their dependencies. It does not claim exclusive authorship or ownership of upstream code. Exact submitted files and content-addressed snapshots are preserved byte-for-byte so their evidence hashes remain useful.

- Challenge interfaces, baseline-derived portions and problem descriptions originate from [Open Kernel Challenge](https://openkernelchallenge.dev/). Obtain the authoritative harness and current terms through its CLI/site.
- SGLang-derived baseline portions reference commit `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`. [SGLang](https://github.com/sgl-project/sglang) uses Apache-2.0; retain applicable upstream notices.
- Embedded FlashInfer grouped GEMM implementations derive from [FlashInfer](https://github.com/flashinfer-ai/flashinfer), Apache-2.0. Existing headers/notices are retained inside embedded source strings.
- Hopper mixed-input mainloop, scheduler helpers and the weight-interleaving permutation derive from NVIDIA TensorRT-LLM sources distributed inside the pinned FlashInfer image. The CUDA copies retain NVIDIA's 2020–2023 notice; the interleaver derives from a file bearing Copyright (c) 2020–2025, NVIDIA CORPORATION, licensed under Apache-2.0. The Python adaptation is `work/nvfp4_hopper_probe_helpers.py`.
- CUTLASS/CuTe, Triton, PyTorch and other packages are external dependencies, not redistributed package installations. Their respective licenses remain applicable. The pinned container is subject to its own upstream terms.

Apache-2.0 text is included in `licenses/Apache-2.0.txt`. Original modifications are licensed under Apache-2.0; this grant covers only rights in those modifications and does not replace upstream rights, notices or challenge terms. Failed and partial variants are research artifacts, with no warranty of correctness or performance.

Modified components include custom Triton/CUDA kernels, shape dispatch, lossless preparation/repacking, fusion boundaries, launch choices and diagnostic harnesses. The Hopper changes are described separately in `docs/hopper-prototype.md`.
