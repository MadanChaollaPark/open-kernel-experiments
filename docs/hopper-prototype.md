# Unfinished Hopper NVFP4 prototype

The pinned mixed-input backend uses group-32 scales and a UE8M0 conversion, so it cannot directly represent the challenge's group-16 E4M3 scales. The experiment introduced a separate `OkcNvfp4Mainloop` with group size 16, E4M3 conversion preserving mantissas, and a folded scale layout. It uses BF16 activations, FP32 tensor-core accumulation and a 64×64×128 tile with two stages.

The installed upstream modules were not patched. The standalone CUDA source retains upstream NVIDIA notices. Modifications include the named mainloop, scale layout/conversion, namespaced scheduler helper and GPU metadata/descriptor construction. The Python weight interleaver follows the NVIDIA packed-bit permutation; it does not change stored weight precision.

Evidence:

- [Mainloop compilation](../results/probes/nvfp4-hopper-precompile.json): succeeded.
- [GEMM compilation](../results/probes/nvfp4-hopper-gemm-precompile.json): succeeded.
- [Scale layout](../results/probes/nvfp4-hopper-scale-probe.json): 16,384 values, zero bit mismatches.
- [Small grouped GEMM](../results/probes/nvfp4-hopper-gemm-probe.json): five shapes and five fresh-input CUDA graph checks passed, relative L2 below 0.01 against FP32.

The last probe completed before the stop request was processed. No further optimization was launched. This demonstrates limited correctness only: no full MoE integration, challenge validation, performance measurement or winning submission exists for this prototype.

The `work/test_nvfp4_hopper_*` and precompile scripts are historical scripts. They assume the original local tooling/cache/ledger layout. Use them as implementation references; the general replay wrapper does not run these independent probes.
