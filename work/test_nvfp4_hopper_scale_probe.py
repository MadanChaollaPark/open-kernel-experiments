"""Bounded layout diagnostic, not a challenge submission."""
import json
import pathlib
import subprocess
import time

import modal

app = modal.App('okc-muse-experiments')
image = modal.Image.from_registry('lmsysorg/sglang@sha256:8cfc5a72cb296771f5f8a4a4688cb6d05685610a830cbb616fa4ab0eb7bb2b0a').env({'TVM_FFI_CACHE_DIR':'/okc-compile-cache/tvm-ffi','TVM_FFI_CUDA_ARCH_LIST':'9.0a','MAX_JOBS':'2'})
cache = modal.Volume.from_name('okc-coul-sglang-build-cache')

@app.function(image=image, gpu='H200', cpu=(4,4), memory=(8192,8192), timeout=900, startup_timeout=300, max_containers=1, retries=0, scaledown_window=2, single_use_containers=True, volumes={'/okc-compile-cache':cache})
def diagnostic(source):
    import importlib.util
    import sys
    import tempfile
    import torch
    path=pathlib.Path(tempfile.mkdtemp())/'probe.py'
    path.write_text(source)
    spec=importlib.util.spec_from_file_location('probe',path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules['probe']=mod
    spec.loader.exec_module(mod)
    module=mod._warp_module()
    torch.manual_seed(4381)
    # Each stage has [64 output channels, 8 independent E4M3 scale groups].
    finite=torch.tensor([i for i in range(256) if (i&127)!=127],device='cuda',dtype=torch.uint8)
    logical=finite[torch.arange(1024,device='cuda')%finite.numel()].reshape(2,64,8)
    packed=torch.empty((2,512),device='cuda',dtype=torch.uint8)
    m=torch.arange(64,device='cuda')[:,None];g=torch.arange(8,device='cuda')[None,:]
    offsets=(m%32)*16+(m//32)*8+g
    packed[:,offsets.reshape(-1)]=logical.reshape(2,-1)
    out=torch.empty((2,64,128),device='cuda',dtype=torch.float32)
    module.probe_scale_layout(packed,out)
    expected=logical.view(torch.float8_e4m3fn).float().repeat_interleave(16,dim=-1)
    mismatch=(out.view(torch.int32)!=expected.view(torch.int32)).sum().item()
    cache.commit()
    return {'gpu':torch.cuda.get_device_name(),'values':out.numel(),'bit_mismatches':mismatch,'all_passed':mismatch==0,'mainloop_shared_bytes':module.mainloop_shared_bytes(),'kernel_shared_bytes':module.kernel_shared_bytes(),'scope':'CUTE folded E4M3 scale layout for all64channels,8groups and2stages; everyK16group independently scaled. Not a GEMM correctness or performance result.'}

@app.local_entrypoint()
def main():
    subprocess.run(['work/tooling/venv/bin/python','work/reconcile_budget.py'],check=True)
    apps=json.loads(subprocess.check_output(['work/tooling/venv/bin/modal','app','list','--json']))
    if any(a['app_id']!=app.app_id and a['description']=='okc-muse-experiments' and (a['state']!='stopped' or a['tasks']!='0') for a in apps):
        raise RuntimeError('Another GPU experiment is active')
    p=pathlib.Path('outputs/budget.json');ledger=json.loads(p.read_text())
    if sum(r['reserved_usd'] for r in ledger['reservations'])+3>ledger['cap_usd']:
        raise RuntimeError('Budget exhausted')
    ledger['reservations'].append({'label':'nvfp4-hopper-scale-probe','reserved_usd':3,'started_at':time.time()})
    p.write_text(json.dumps(ledger,indent=2))
    result=diagnostic.remote(pathlib.Path('work/nvfp4_hopper_probe_module.py').read_text())
    pathlib.Path('outputs/nvfp4-hopper-scale-probe.json').write_text(json.dumps(result,indent=2))
    print('ALL PASSED',result['all_passed'])
