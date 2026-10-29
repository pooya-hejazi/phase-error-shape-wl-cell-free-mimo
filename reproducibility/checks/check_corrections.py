"""Small independent checks for the corrected estimand, batching and feasibility."""
from pathlib import Path
from types import SimpleNamespace
import json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
(ROOT/'generated/checks').mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'reproducibility/calibrated_network'))
import run_calibrated_network as net
from shape_estimators import widely_linear_mmse_shape, clmmse_shape
sys.path.insert(0,str(ROOT/'analysis'))
from calibration_law import project

cfg=SimpleNamespace(users=1,payload_noise_variance=1.,pilot_symbols=4,coherence_symbols=200)
h=np.array([[[1.]],[[3.]]]); v=np.ones_like(h)
expected=.98*np.log2(3.)
assert np.allclose(net.uatf(h,v,cfg),expected)
old=.98*np.mean(np.log2(1+np.array([1.,9.])))
assert abs(old-expected)>.1
try: net.uatf(h[0],v[0],cfg)
except ValueError: pass
else: raise AssertionError('One-frame input must not masquerade as UatF')

config=net.ScenarioConfig(access_points=2,antennas_per_ap=4,users=4,pilot_symbols=1,
                         coherence_symbols=200,pilot_assignment='balanced_random')
top=net.generate_topology(config,seed=103)
truth,obs=net.make_frame(top,config,'uniform',107)
ma,ca=net.calibrate_chains('uniform',1024,109,2)
assert ma.shape==(2,) and ca.shape==(2,) and ma[0]!=ma[1]
gaps=[]
for wl in (False,True):
    got=net.estimate(truth,obs,top,config,ma,ca if wl else 0.,wl)
    for ap in range(2):
        users=np.flatnonzero(top.pilot_indices==0)
        model=net.ShapePilotGroupModel(los=top.tracked_los[ap][:,users],
          covariances=top.diffuse_covariances[ap,users],beta=np.full(4,config.pilot_coefficient,dtype=complex),
          noise_covariance=config.pilot_noise_variance*np.eye(4),m1=np.full(4,ma[ap]),
          m2=np.full(4,ma[ap]**2+(ca[ap] if wl else 0.)))
        for j,u in enumerate(users):
            ref=(widely_linear_mmse_shape if wl else clmmse_shape)(model,obs[ap,:,0],j)
            gaps.append(float(np.max(np.abs(got[ap,:,u]-ref))))
assert max(gaps)<1e-10
# Unit-norm and unnormalized pilot despreading must give identical estimates.
beta, variance, pseudo, noise, y = 2., .7, -.2, 1., .3+.2j
def scalar_estimate(beta,noise,y):
    cy=beta**2*variance+noise
    a=beta*(variance*cy-beta**2*pseudo**2)/(cy**2-beta**4*pseudo**2)
    b=beta*pseudo*noise/(cy**2-beta**4*pseudo**2)
    return a*y+b*y.conjugate()
assert abs(scalar_estimate(beta,noise,y)-scalar_estimate(2*beta,4*noise,2*y))<1e-14
for m,c in [(1.1,-.2),(-.1,.3),(.9,-1),(.9,.1)]:
    mm,cc,active=project(m,c)
    assert 0<mm<1 and -(1-mm*mm)<=cc<=0 and active
result={'two_frame_uatf':float(net.uatf(h,v,cfg)[0]),'expected':float(expected),
        'old_mean_instantaneous_rate':float(old),'max_batch_reference_gap':max(gaps),
        'independent_ap_calibration':True,'feasible_clipping':True}
(ROOT/'generated/checks/correction_checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
