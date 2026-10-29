"""Compare actual-law moments and raw joint calibration covariance to analytic values."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.special import ive
ROOT=Path(__file__).resolve().parents[2]
(ROOT/'generated/checks').mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(ROOT/'analysis'))
from calibration_law import draw_phases,SIGMA_W2
from phase_laws import uniform_delta_for_m1,vm_kappa_for_m1
m=.9; w=SIGMA_W2; count=4000; frames=1024; out={}
for li,law in enumerate(('binary','uniform','wrapped_gaussian','von_mises')):
    rng=np.random.default_rng(2026102990+li)
    if law=='binary': moments=np.cos(np.arange(1,5)*np.arccos(m))
    elif law=='uniform': moments=np.sinc(np.arange(1,5)*uniform_delta_for_m1(m)/np.pi)
    elif law=='wrapped_gaussian': moments=m**(np.arange(1,5)**2)
    else:
        k=vm_kappa_for_m1(m);moments=ive(np.arange(1,5),k)/ive(0,k)
    _,m2,m3,m4=moments;c=m2-m*m;v=1-m*m+w
    sigma=np.array([[(v+c)/2,(m3-m)/2-2*m*c],
       [(m3-m)/2-2*m*c,(1+m4)/2+4*m*m*m2-2*m*m3-3*m**4-c*c+2*w*(1-m*m)+w*w]])
    estimates=[];diagnostics=[]
    for start in range(0,count,200):
        eps=draw_phases(law,200*frames,rng).reshape(200,frames)
        z=np.exp(1j*eps)+np.sqrt(w/2)*(rng.normal(size=eps.shape)+1j*rng.normal(size=eps.shape))
        zbar=z.mean(axis=1); centered=z-zbar[:,None]
        estimates.extend(zip(zbar.real,np.real(np.sum(centered**2,axis=1))/(frames-1)))
        if start==0:
            truthcenter=z-m
            tests=[(np.real(truthcenter**2),c),(np.real(truthcenter**3),m3-3*m*m2+2*m**3),
              (abs(truthcenter)**4,1-3*m**4+2*m*m*m2+4*w*(1-m*m)+2*w*w)]
            for vals,target in tests:
                se=vals.std()/np.sqrt(vals.size);err=abs(vals.mean()-target)
                assert err<5*se,(law,err,se)
                diagnostics.append({'estimate':float(vals.mean()),'analytic':float(target),'mcse':float(se)})
    measured=frames*np.cov(np.array(estimates).T,ddof=1)
    # Normal-window covariance sampling tolerance plus a finite-window 1/M allowance.
    tolerance=5*np.sqrt((sigma*sigma+np.outer(np.diag(sigma),np.diag(sigma)))/(count-1))+.002
    assert np.all(abs(measured-sigma)<tolerance),(law,measured,sigma)
    out[law]={'circular_moments_1_to_4':moments.tolist(),'central_moment_checks_2_3_abs4':diagnostics,
        'analytic_asymptotic_Sigma':sigma.tolist(),'raw_empirical_M_cov':measured.tolist(),
        'windows':count,'frames':frames,'passed':True}
(ROOT/'generated/checks/calibration_moment_checks.json').write_text(json.dumps(out,indent=2))
print('Actual-law moments 2, 3, |.|^4 and raw joint covariance passed for all four laws.')
