"""Bounded H200 experiments. Submission source is passed unchanged."""
import json
import pathlib
import time
import modal

ROOT = pathlib.Path(__file__).resolve().parent
IMAGE = 'lmsysorg/sglang@sha256:8cfc5a72cb296771f5f8a4a4688cb6d05685610a830cbb616fa4ab0eb7bb2b0a'
app = modal.App('okc-muse-experiments')
image = modal.Image.from_registry(IMAGE)
compile_cache=modal.Volume.from_name('okc-coul-sglang-build-cache',create_if_missing=True)

@app.function(image=image, gpu='H200', cpu=(4,4), memory=(8192,8192),
              timeout=900, startup_timeout=300, max_containers=1,
              scaledown_window=2, retries=0, single_use_containers=True,
              volumes={'/okc-compile-cache':compile_cache})
def experiment(files, candidates, cases, extras, repeats=30):
    import importlib.util, sys, tempfile, statistics, math, platform
    import shutil
    cache_dir=pathlib.Path('/root/.cache/sglang')
    if cache_dir.exists() and not cache_dir.is_symlink():
        shutil.copytree(cache_dir,'/okc-compile-cache',dirs_exist_ok=True)
        shutil.rmtree(cache_dir)
    if not cache_dir.exists(): cache_dir.symlink_to('/okc-compile-cache',target_is_directory=True)
    # The package compares resolved dependency paths against its build directory.
    # Give it a canonical cache root so generated temporary sources are excluded.
    import os
    os.environ['SGLANG_JIT_CACHE_DIR']=str(pathlib.Path('/okc-compile-cache/jit').resolve())
    os.environ['TRITON_CACHE_DIR']=str(pathlib.Path('/okc-compile-cache/triton').resolve())
    os.environ['FLASHINFER_WORKSPACE_BASE']='/okc-compile-cache/flashinfer-workspace'
    os.environ['TVM_FFI_CACHE_DIR']='/okc-compile-cache/tvm-ffi'
    os.environ['TVM_FFI_CUDA_ARCH_LIST']='9.0a'
    import torch, triton
    start = time.monotonic()
    root = pathlib.Path(tempfile.mkdtemp())
    for name, source in files.items():
        (root / name).write_text(source)
    sys.path.insert(0, str(root))
    import reference, baseline
    torch.backends.cuda.matmul.allow_tf32 = False
    weights = reference.generate_weights(5151) if hasattr(reference,'generate_weights') else None
    weight_fields={}
    if weights is not None and hasattr(reference,'weights_fields'):
        print('Preparing official router load fields',flush=True)
        weight_fields=reference.weights_fields(weights,cases,5151)
        if weight_fields:
            weights=reference.generate_weights(5151,**weight_fields)
        print('Prepared router load fields',weight_fields,flush=True)
    case_field_cache={}
    def make_data(cfg):
        key=json.dumps(cfg,sort_keys=True)
        # Optional additional validation inputs. Scored benchmark cases do not
        # set this field; the official reference remains unchanged.
        input_cfg={k:v for k,v in cfg.items() if k!='routing_pattern'}
        if key not in case_field_cache:
            case_field_cache[key]=reference.case_fields(input_cfg,weights) if hasattr(reference,'case_fields') else {}
        data=reference.generate_input(**{**input_cfg,**case_field_cache[key]})
        pattern=cfg.get('routing_pattern')
        if pattern is not None:
            ids=data['topk_ids'];width=ids.shape[-1]
            route=torch.arange(width,device=ids.device,dtype=ids.dtype)
            if pattern=='all_nonlocal':route=route+32
            elif pattern=='one_local':route=torch.where(route==0,route,route+31)
            elif pattern!='all_local':raise ValueError('Unknown validation routing pattern')
            ids.copy_(route.expand_as(ids))
        return data
    base_state = baseline.prepare(weights) if hasattr(baseline,'prepare') else None
    def invoke(mod,data,state):
        try:
            return mod.kernel_fn(data,state) if weights is not None else mod.kernel_fn(data)
        except Exception as error:
            parts=[f'{type(error).__name__}: {error}']
            cause=error.__cause__
            for _ in range(8):
                if cause is None:break
                parts.append(f'{type(cause).__name__}: {cause}');cause=cause.__cause__
            raise RuntimeError(mod.__name__+': '+chr(10).join(parts)) from None
    modules = {}
    setup_failures=[]
    for name, source in candidates.items():
        path = root / (name + '.py')
        path.write_text(source)
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
            modules[name] = (mod, mod.prepare(weights) if hasattr(mod,'prepare') else None)
        except Exception as error:
            message=f'{type(error).__name__}: {error}'
            setup_failures.append({'candidate':name,'case':{'phase':'import/prepare'},'ok':False,'message':message})
            print('SETUP ERROR',name,message,flush=True)
    result = {'environment': {'gpu':torch.cuda.get_device_name(), 'torch':str(torch.__version__),
              'triton':triton.__version__, 'python':platform.python_version(), 'cuda':torch.version.cuda},
              'validation':setup_failures, 'timings':[], 'weight_fields':weight_fields, 'case_fields':case_field_cache}
    print(json.dumps(result['environment']), flush=True)
    failed=set()
    for cfg in extras + cases:
        data = make_data(cfg)
        for name,(mod,state) in modules.items():
            if name in failed: continue
            print('CHECK',name,cfg,flush=True)
            try:
                out = invoke(mod,data,state)
            except Exception as error:
                result['validation'].append({'candidate':name,'case':cfg,'ok':False,'message':str(error)})
                failed.add(name)
                print('CANDIDATE ERROR',name,str(error),flush=True)
                continue
            if cfg.get('routing_pattern')=='all_nonlocal':
                # The official checker deliberately declines empty local
                # routing. For this additional edge case the mathematical
                # output is exactly zero; use a separate strict assertion.
                ok=(tuple(out.shape)==tuple(data['hidden_states'].shape)
                    and out.dtype==torch.bfloat16
                    and bool(torch.isfinite(out).all())
                    and int(torch.count_nonzero(out))==0)
                msg='Additional empty-route check: expected exact zero BF16 output'
            elif weights is None:
                ok,msg = reference.check_implementation(data,reference.collect(data,out))
            else:
                checked=reference.collect(data,out) if hasattr(reference,'collect') else out
                ok,msg = reference.check_implementation(data,checked,weights)
            result['validation'].append({'candidate':name,'case':cfg,'ok':ok,'message':msg})
            if not ok:
                print('VALIDATION FAIL', name,cfg,msg,flush=True)
                failed.add(name)
        print('Completed checks for case',cfg,flush=True)
        if len(failed)==len(modules): break
    print('Validation complete. Excluded:',sorted(failed),flush=True)
    if len(failed)==len(modules):
        result['elapsed_s']=time.monotonic()-start
        compile_cache.commit()
        return json.dumps(result)
    flush = torch.ones(32*1024*1024,device='cuda',dtype=torch.float32)
    flush_total = torch.empty((),device='cuda',dtype=torch.float32)
    def capture(fn):
        for _ in range(3): fn()
        torch.cuda.synchronize()
        a,b=torch.cuda.Event(enable_timing=True,external=True),torch.cuda.Event(enable_timing=True,external=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            torch.sum(flush,dim=0,out=flush_total)
            a.record()
            out=fn()
            b.record()
        return (graph,a,b),out
    def measure(bundle):
        graph,a,b=bundle
        graph.replay()
        b.synchronize()
        return a.elapsed_time(b)*1000
    empty,empty_output=capture(lambda:torch.empty((1,),device='cuda'))
    result['calibration_us']=statistics.median([measure(empty) for _ in range(30)])
    result['timing_method']='Events inside a replayed CUDA graph, after a 128MiB cache read; subtract measured empty-graph event interval.'
    print('Measured fixed timing cost',result['calibration_us'],flush=True)
    for cfg in cases:
        data=make_data(cfg)
        bg,bo=capture(lambda:invoke(baseline,data,base_state))
        for name,(mod,state) in modules.items():
            if name in failed: continue
            cg,co=capture(lambda:invoke(mod,data,state))
            bs,cs=[],[]
            for i in range(repeats):
                if i%2: c,b=measure(cg),measure(bg)
                else: b,c=measure(bg),measure(cg)
                bs.append(b);cs.append(c)
            row={'candidate':name,'case':cfg,'baseline_us':statistics.median(bs),
                 'candidate_us':statistics.median(cs),'baseline_samples':bs,'candidate_samples':cs}
            result['timings'].append(row)
            print(name,cfg,round(row['baseline_us'],2),round(row['candidate_us'],2),flush=True)
    # Diagnostic profiling runs after timing and is never part of submission code.
    if result['timings']:
        case_bests=[]
        for cfg in cases:
            rows=[r for r in result['timings'] if r['case']==cfg]
            if rows:
                fastest=min(rows,key=lambda r:r['candidate_us'])
                case_bests.append(fastest)
        chosen=max(case_bests,key=lambda r:r['candidate_us']*r['case'].get('weight',1))
        data=make_data(chosen['case'])
        winner=chosen['candidate'];mod,state=modules[winner]
        result['profile_bottleneck_case']={'candidate_name':winner,'case':chosen['case']}
        try:
            for label,profile_mod,profile_state in [('candidate',mod,state),('baseline',baseline,base_state)]:
                pg,po=capture(lambda:invoke(profile_mod,data,profile_state))
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                    for _ in range(4):pg[0].replay()
                    torch.cuda.synchronize()
                result['profile_bottleneck_case'][label+'_table']=prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=15)
        except Exception as error:
            result['profile_error']=str(error)
    result['algorithm_counts']={}
    for _,state in modules.values():
        runner=state.get('runner') if isinstance(state,dict) else None
        for key,value in getattr(runner,'_algo_cache',{}).items():
            result['algorithm_counts'][str(key)]=int(value[1])
    # Compiler diagnostics belong to the experiment harness, never submission code.
    result['compiler_diagnostics']=[]
    for name,(mod,state) in modules.items():
        for fn_name,fn in vars(mod).items():
            if not hasattr(fn,'device_caches'):continue
            try:
                for entry in fn.device_caches.values():
                    for compiled in entry[0].values():
                        result['compiler_diagnostics'].append({'candidate':name,'function':fn_name,'registers':getattr(compiled,'n_regs',None),'spills':getattr(compiled,'n_spills',None),'shared':getattr(getattr(compiled,'metadata',None),'shared',None)})
            except Exception as error:
                result['compiler_diagnostics'].append({'candidate':name,'function':fn_name,'diagnostic_error':str(error)})
    if any('warp_mma_run' in source for source in candidates.values()):
        # Static CUDA resource inspection happens after all timing and correctness.
        import subprocess
        result['cuda_resource_usage']={}
        for binary in pathlib.Path('/okc-compile-cache/tvm-ffi').rglob('*warp_mma*.so'):
            try:
                result['cuda_resource_usage'][str(binary)]=subprocess.check_output(
                    ['cuobjdump','--dump-resource-usage',str(binary)],text=True,timeout=30)
            except Exception as error:
                result['cuda_resource_usage'][str(binary)]=str(error)
    result['elapsed_s']=time.monotonic()-start
    compile_cache.commit()
    return json.dumps(result)

@app.local_entrypoint()
def main(candidate: str='work/kernels/muse_bf16_proj/submission.py', label: str='initial', repeats: int=30, problem_name: str='muse_bf16_proj', layer_filter: str='', extras_file: str=''):
    # Reject undefined generated names before reserving or launching a GPU.
    import subprocess
    subprocess.run([str(ROOT/'tooling/venv/bin/ruff'),'check','--select','F821',candidate],check=True)
    # Reserve worst-case spend before launching. Reservations are never refunded
    # from function timing alone: startup and image costs also consume the cap.
    ledger_path=ROOT.parent/'outputs/budget.json'
    ledger=json.loads(ledger_path.read_text())
    reserved=sum(x['reserved_usd'] for x in ledger['reservations'])
    if reserved+3>ledger['cap_usd']: raise RuntimeError('Compute budget exhausted')
    ledger['reservations'].append({'label':label,'reserved_usd':3,'started_at':time.time()})
    ledger_path.write_text(json.dumps(ledger,indent=2))
    problem=json.loads((ROOT/f'challenge/api_v1_problems_{problem_name}.json').read_text())
    extras=[{'m':m,'layer':layer,'seed':700+i,'dist':dist} for i,(m,layer,dist) in enumerate([
       (1,'qkv_gate','normal'),(7,'pooler','normal'),(33,'o','normal'),(130,'unpooler','normal'),
       (5,'unpooler','normal'),(8,'router','normal'),(9,'router','outliers'),
       (512,'qkv_gate','outliers'),(1000,'unpooler','outliers')])]
    if problem_name=='muse_attention_decode':
        extras=json.loads((ROOT/'attention_extra_cases.json').read_text())
    elif problem_name=='muse_decode_layer':
        extras=json.loads((ROOT/'decode_extra_cases.json').read_text())
    elif problem_name=='muse_prefill_layer':
        extras=[dict(batch=b,ctx=c,layer=l,q_len=q,page=1,dist=d,seed=2100+i) for i,(b,c,l,q,d) in enumerate([(1,0,'sliding',300,'normal'),(1,3000,'full',753,'normal'),(2,2500,'sliding',753,'normal'),(3,5000,'full',200,'outliers'),(1,4096,'sliding',4096,'normal')])]
    elif problem_name=='muse_moe_w4a16':
        extras=[{'bs':bs,'sharding':'ep8','seed':1800+i,'dist':dist} for i,(bs,dist) in enumerate([(4,'normal'),(33,'normal'),(257,'normal'),(128,'skewed'),(1,'normal')])]
    if extras_file:
        extras.extend(json.loads(pathlib.Path(extras_file).read_text()))
    candidate_path=pathlib.Path(candidate)
    candidates={p.stem:p.read_text() for p in sorted(candidate_path.glob('*.py'))} if candidate_path.is_dir() else {label:candidate_path.read_text()}
    cases=problem['benchmarks']
    if layer_filter:
        cases=[c for c in cases if c.get('layer')==layer_filter]
        extras=[c for c in extras if c.get('layer')==layer_filter]
    result=json.loads(experiment.remote(problem['files'],candidates,cases,extras,repeats))
    result['case_scope']=('layer='+layer_filter) if layer_filter else 'full problem suite'
    result['image']=IMAGE
    result['sources']=candidates
    result['problem']=problem_name
    output=ROOT.parent/'outputs'/f'{label}.json'
    output.write_text(json.dumps(result,indent=2))
    print('Saved',output)
