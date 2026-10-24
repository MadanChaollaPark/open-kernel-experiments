import collections,json,math,pathlib,statistics,sys
d=json.load(open(sys.argv[1])); overhead=d.get('calibration_us',4)
rows=collections.defaultdict(dict)
for r in d['timings']:
    b=[max(x-overhead,.1) for x in r['baseline_samples']]
    c=[max(x-overhead,.1) for x in r['candidate_samples']]
    ratio=statistics.median([math.sqrt(b[i]*b[i+1]/c[i]/c[i+1]) for i in range(0,len(b)-1,2)])
    cfg=r['case'];key=(cfg.get('m',cfg.get('batch',cfg.get('bs'))),cfg.get('layer','moe'),cfg.get('q_len',1))
    if d.get('problem')=='muse_prefill_layer':key=key+(cfg.get('ctx'),)
    rows[key][r['candidate']]={'ratio':ratio,'base':statistics.median(b),'case':cfg}
def score(choices):
    loads=collections.defaultdict(lambda:[0,0])
    for key,name in choices.items():
        row=rows[key][name];cfg=row['case']
        base=rows[key].get('control',row)['base']
        loads[cfg['load']][0]+=base*cfg['weight']
        loads[cfg['load']][1]+=base/row['ratio']*cfg['weight']
    return math.exp(statistics.mean(math.log(b/c) for b,c in loads.values()))
names=set.intersection(*(set(r) for r in rows.values())) if rows else set()
print('Timing:',d.get('timing_method','outside-graph events'),'subtraction:',overhead,'us')
for val,name in sorted([(score({k:n for k in rows}),n) for n in names],reverse=True): print(f'{name:30} {val:.4f}x')
best={k:max(r,key=lambda n:r[n]['ratio']) for k,r in rows.items()}
for k in best:
    control=next((n for n in ('control','hybrid_control') if n in rows[k]),None)
    if control and rows[k][best[k]]['ratio']<1.03*rows[k][control]['ratio']:best[k]=control
if best: print('Composite estimate (requires separate validation):',round(score(best),4))
for key,name in best.items():print(key,name,round(rows[key][name]['ratio'],3))
