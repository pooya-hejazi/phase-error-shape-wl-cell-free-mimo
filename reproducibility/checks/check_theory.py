"""Independent analytical, sampling, and archived-replay checks for the final audit."""
from pathlib import Path
import importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
(ROOT/'generated/checks').mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'analysis'))
from phase_laws import law_rows
sys.path.insert(0,str(ROOT/'reproducibility/expanded_run_v02/run'))
from shape_model import ShapePilotGroupModel
from shape_estimators import widely_linear_mmse_shape,clmmse_shape

rng=np.random.default_rng(2026102907)
ell=np.array([[1+.2j,.4-.3j],[.1j,.7+.2j]])
beta=np.array([1.2,.9]); m=.9
c=law_rows(m)['uniform']['c'] if isinstance(law_rows(m),dict) else None
if c is None:
    c=next(row['c'] for row in law_rows(m) if row['law']=='uniform')
delta=0.786683072049212
from scipy.optimize import brentq
delta=brentq(lambda x:np.sinc(x/np.pi)-m,1e-8,3.14)
cov=np.array([np.eye(2)*.05,np.eye(2)*.08]); noise=np.eye(2)*.01
V=cov+np.array([(1-m*m)*np.outer(ell[:,i],ell[:,i].conj()) for i in range(2)])
P=np.array([c*np.outer(ell[:,i],ell[:,i]) for i in range(2)])
Cy=noise+np.einsum('i,ijk->jk',beta**2,V); Py=np.einsum('i,ijk->jk',beta**2,P)
Caug=np.block([[Cy,Py],[Py.conj(),Cy.conj()]])
cross=np.concatenate([beta[0]*V[0],beta[0]*P[0]],axis=1)
W=np.linalg.solve(Caug.T,cross.T).T
W2=np.linalg.solve(Cy.T,(beta[0]*V[0]).T).T
model=ShapePilotGroupModel(los=ell,covariances=cov,beta=beta.astype(complex),noise_covariance=noise,m1=np.full(2,m),m2=np.full(2,m*m+c))
y=rng.normal(size=2)+1j*rng.normal(size=2); muy=ell@(beta*m)
expected=ell[:,0]*m+W@np.r_[y-muy,y.conj()-muy.conj()]
assert np.max(abs(expected-widely_linear_mmse_shape(model,y,0)))<1e-12
samples=200000
phase=rng.uniform(-delta,delta,size=(samples,2))
g=(rng.normal(size=(samples,2,2))+1j*rng.normal(size=(samples,2,2)))*np.sqrt(np.array([.05,.08])[None,None,:]/2)
h=ell[None,:,:]*np.exp(1j*phase[:,None,:])+g
y=np.einsum('snk,k->sn',h,beta)+(rng.normal(size=(samples,2))+1j*rng.normal(size=(samples,2)))*np.sqrt(.01/2)
yc=y-muy
hat1=ell[:,0]*m+np.concatenate([yc,yc.conj()],axis=1)@W.T
hat2=ell[:,0]*m+yc@W2.T
risk1=np.sum(abs(h[:,:,0]-hat1)**2,axis=1); risk2=np.sum(abs(h[:,:,0]-hat2)**2,axis=1)
analytic1=float(np.trace(V[0]-W@cross.conj().T).real)
analytic2=float(np.trace(V[0]-W2@(beta[0]*V[0]).conj().T).real)
assert abs(risk1.mean()-analytic1)<5*risk1.std()/np.sqrt(samples)
assert abs(risk2.mean()-analytic2)<5*risk2.std()/np.sqrt(samples)
assert analytic1<analytic2
# A proper target does not imply a proper pilot group.
mixed=ShapePilotGroupModel(los=ell,covariances=cov,beta=beta.astype(complex),noise_covariance=noise,m1=np.full(2,m),m2=np.array([m*m,m*m+c]))
probe=np.array([1+.7j,-.6+.3j])
mixed_gap=float(np.linalg.norm(widely_linear_mmse_shape(mixed,probe,0)-clmmse_shape(mixed,probe,0)))
assert mixed_gap>1e-6
# Check the claimed one-sided minimax endpoint, including full complex coefficients.
v=.24; s=.01; bta=1.; pmin=.05; pmax=.19; cy=v+s
def risk(a,b,p):
    return v+(abs(a)**2+abs(b)**2)*cy-2*v*np.real(a)-2*np.real(np.conj(b)*p)+2*np.real(a*np.conj(b)*p)
d=cy**2-pmin**2; a=(v*cy-pmin**2)/d; b=-pmin*s/d
vals=np.array([risk(a,b,p) for p in np.linspace(-pmax,-pmin,1001)])
assert np.argmax(vals)==1000 and vals.max()<v*s/cy
assert 2*b*(a-1)>0

spec=importlib.util.spec_from_file_location('replay',ROOT/'reproducibility/v08_run/replay.py')
replay=importlib.util.module_from_spec(spec); spec.loader.exec_module(replay)
stored=json.loads(replay.STORED.read_text()); fresh=json.loads(replay.FRESH_COPY.read_text())
mis,diff=replay.compare(replay.scientific_view(stored),replay.scientific_view(fresh))
assert not mis
mut=json.loads(json.dumps(replay.scientific_view(fresh)))
mut['records'][0]['mse_B1']*=1.01
assert replay.compare(replay.scientific_view(stored),mut)[0]
result={'vector_complex_formula_max_error':float(np.max(abs(expected-widely_linear_mmse_shape(model,probe,0)))) if False else 'passed against direct solve',
    'samples':samples,'B1_analytic':analytic1,'B1_MC':float(risk1.mean()),'B1_MC_se':float(risk1.std()/np.sqrt(samples)),
    'B2_analytic':analytic2,'B2_MC':float(risk2.mean()),'B2_MC_se':float(risk2.std()/np.sqrt(samples)),
    'proper_target_improper_group_B1_B2_gap':mixed_gap,'minimax_endpoint_check':True,
    'archived_replay_record_count':len(fresh['records']),'archived_replay_max_relative_diff':diff,'mutation_detected':True}
(ROOT/'generated/checks/theory_checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
