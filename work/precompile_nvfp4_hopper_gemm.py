import modal,pathlib,json
app=modal.App('okc-nvfp4-hopper-gemm-precompile')
image=modal.Image.from_registry('lmsysorg/sglang@sha256:8cfc5a72cb296771f5f8a4a4688cb6d05685610a830cbb616fa4ab0eb7bb2b0a').env({'TVM_FFI_CACHE_DIR':'/okc-compile-cache/tvm-ffi','TVM_FFI_CUDA_ARCH_LIST':'9.0a','MAX_JOBS':'2'})
cache=modal.Volume.from_name('okc-coul-sglang-build-cache')
@app.function(image=image,cpu=(4,4),memory=(8192,8192),timeout=300,startup_timeout=120,max_containers=1,retries=0,scaledown_window=2,volumes={'/okc-compile-cache':cache})
def compile_source(source):
 import importlib.util,time,traceback
 from tvm_ffi.cpp import load_inline
 root=pathlib.Path(next(iter(importlib.util.find_spec('flashinfer').submodule_search_locations)))/'data'
 start=time.monotonic()
 try:
  module=load_inline('okc_nvfp4_hopper_gemm_v1',cuda_sources=source,backend='cuda',extra_cuda_cflags=['-std=c++17','-O3','--expt-relaxed-constexpr','--expt-extended-lambda','-gencode=arch=compute_90a,code=sm_90a'],extra_include_paths=[str(root/'include'),str(root/'csrc'),str(root/'cutlass/include'),str(root/'cutlass/tools/util/include'),str(root/'csrc/nv_internal/tensorrt_llm/cutlass_extensions/include'),str(root/'csrc/nv_internal')])
  result={'compiled':True,'elapsed_s':time.monotonic()-start,'mainloop_shared_bytes':module.mainloop_shared_bytes(),'kernel_shared_bytes':module.kernel_shared_bytes()};cache.commit();return result
 except Exception as e:return {'compiled':False,'elapsed_s':time.monotonic()-start,'error':str(e),'traceback':traceback.format_exc()}
@app.local_entrypoint()
def main():
 r=compile_source.remote(pathlib.Path('work/nvfp4_hopper_gemm.cu').read_text());pathlib.Path('outputs/nvfp4-hopper-gemm-precompile.json').write_text(json.dumps(r,indent=2));print({k:v for k,v in r.items() if k not in ('error','traceback')})
