"""Extend the paired downlink expectations; preserve the historical 16-frame output."""
from pathlib import Path
import sys,json,os
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'reproducibility/downlink'))
import run_downlink as dl
cfg=dl.ScenarioConfig(access_points=32,antennas_per_ap=4,users=16,pilot_symbols=4,
    coherence_symbols=200,area_side=1.,minimum_distance=.03,pathloss_exponent=3.7,
    rician_kappa=20.,pilot_assignment='balanced_random',pilot_noise_variance=1.,payload_noise_variance=1.)
out=ROOT/'reproducibility/downlink'/os.environ.get('DOWNLINK_OUTPUT','downlink_convergence.json')
assert not out.exists()
records=[]
for t in range(32):
    top=dl.generate_topology(cfg,seed=dl.BASE_SEED+t)
    for code,law in [('U','uniform'),('G','wrapped_gaussian'),('B','binary')]:
        spec=dl.shape_spec(law,.9)
        sums={k:(np.zeros(16,complex),np.zeros(16),np.zeros((16,16))) for k in ('B1','B2')}
        for r in range(128):
            truth,est=dl.estimates(cfg,top,spec,dl.BASE_SEED+7919*r+t)
            for k in sums:
                dl.accumulate(truth,est[k],*sums[k])
            if r+1 in (16,32,64,128):
                records.append({'topology':t,'law':code,'frames':r+1,
                    **{k:dl.finalise(*sums[k],r+1).tolist() for k in sums}})
    if (t+1)%8==0: print('downlink topologies',t+1,flush=True)
summary={}
for n in (16,32,64,128):
    per={}
    for law in ('U','G','B'):
        rows=[v for v in records if v['law']==law and v['frames']==n]
        for k in ('B1','B2'): per[k+'_'+law]=np.array([np.mean(v[k]) for v in rows])
    per['CAL_B']=per['B1_B']  # unused legacy field required by the existing summary helper
    paired=dl.paired_summary(per); paired.pop('CAL_minus_B1_B')
    summary[n]={'means':{k:float(v.mean()) for k,v in per.items() if k!='CAL_B'},'paired':paired}
historical=json.loads((ROOT/'reproducibility/downlink/results.json').read_text())
assert all(abs(v-historical['summary'][k]['mean_se'])<1e-12 for k,v in summary[16]['means'].items())
out.write_text(json.dumps({'topologies':32,'frames':128,'seed':dl.BASE_SEED,'summary':summary,'records':records},indent=2))
print(json.dumps(summary,indent=2))
