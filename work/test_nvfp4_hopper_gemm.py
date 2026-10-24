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
def diagnostic(source, helpers):
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
    namespace={}
    exec(helpers,namespace)
    interleave=namespace['interleave_weights'];fold=namespace['fold_scales']
    torch.manual_seed(4381)
    table=torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],device='cuda',dtype=torch.float32)
    ws=torch.empty(8*1024*1024,device='cuda',dtype=torch.uint8)
    map_ws=torch.empty(4*1024*1024,device='cuda',dtype=torch.uint8)
    metadata=torch.empty(module.nvfp4_meta_bytes(),device='cuda',dtype=torch.uint8)
    results=[]
    for e,m,n,k in [(1,1,64,128),(1,17,128,256),(2,33,256,512),(4,65,128,1024),(2,16,256,4096)]:
        weight=torch.randint(0,256,(e,n,k//2),device='cuda',dtype=torch.uint8)
        # All scale mantissas appear, and adjacent K16 groups differ.
        sf=(torch.arange(e*n*k//16,device='cuda')%48+24).to(torch.uint8).reshape(e,n,k//16)
        x=torch.randn((e,m,k),device='cuda',dtype=torch.bfloat16)
        y=torch.empty((e,m,n),device='cuda',dtype=torch.bfloat16)
        packed=interleave(weight);scales=fold(sf)
        codes=torch.stack((weight&15,weight>>4),dim=-1).reshape(e,n,k).long()
        dequant=table[codes]*sf.view(torch.float8_e4m3fn).float().repeat_interleave(16,dim=-1)
        reference=torch.bmm(x.float(),dequant.transpose(1,2))
        module.run_nvfp4_gemm(packed,scales,x,y,metadata,map_ws,ws)
        torch.cuda.synchronize()
        err=(y.float()-reference).norm()/reference.norm()
        finite=torch.isfinite(y).all().item();passed=finite and err.item()<0.01
        result={'shape':[e,m,n,k],'relative_l2':err.item(),'finite':finite,'passed':passed}
        results.append(result);print(result,flush=True)
        if not passed:break
        # Fresh activations must be recomputed by graph replay.
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):module.run_nvfp4_gemm(packed,scales,x,y,metadata,map_ws,ws)
        x.copy_(torch.randn_like(x));graph.replay();torch.cuda.synchronize()
        reference=torch.bmm(x.float(),dequant.transpose(1,2))
        err=(y.float()-reference).norm()/reference.norm()
        passed=torch.isfinite(y).all().item() and err.item()<0.01
        results.append({'shape':[e,m,n,k],'mode':'graph_fresh_input','relative_l2':err.item(),'passed':passed});print(results[-1],flush=True)
        if not passed:break
    cache.commit()
    return {'gpu':torch.cuda.get_device_name(),'validation':results,'all_passed':len(results)==10 and all(r['passed'] for r in results),'scope':'Independent grouped NVFP4/E4M3group16 times BF16 GEMM against FP32; not yet full MoE or official timing'}

@app.local_entrypoint()
def main():
    subprocess.run(['work/tooling/venv/bin/python','work/reconcile_budget.py'],check=True)
    apps=json.loads(subprocess.check_output(['work/tooling/venv/bin/modal','app','list','--json']))
    if any(a['app_id']!=app.app_id and a['description']=='okc-muse-experiments' and (a['state']!='stopped' or a['tasks']!='0') for a in apps):
        raise RuntimeError('Another GPU experiment is active')
    p=pathlib.Path('outputs/budget.json');ledger=json.loads(p.read_text())
    if sum(r['reserved_usd'] for r in ledger['reservations'])+3>ledger['cap_usd']:
        raise RuntimeError('Budget exhausted')
    ledger['reservations'].append({'label':'nvfp4-hopper-gemm-probe','reserved_usd':3,'started_at':time.time()})
    p.write_text(json.dumps(ledger,indent=2))
    result=diagnostic.remote(pathlib.Path('work/nvfp4_hopper_gemm_probe_module.py').read_text(), pathlib.Path('work/nvfp4_hopper_probe_helpers.py').read_text())
    pathlib.Path('outputs/nvfp4-hopper-gemm-probe.json').write_text(json.dumps(result,indent=2))
    print('ALL PASSED',result['all_passed'])
