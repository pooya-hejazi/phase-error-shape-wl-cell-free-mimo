"""Compare JSON science values, retaining all keys; report numerical differences."""
import argparse,json,math
from pathlib import Path

def differences(a,b,path='',rtol=1e-10,atol=1e-12):
    if isinstance(a,dict) and isinstance(b,dict):
        out=[]
        for k in sorted(a.keys()|b.keys()):
            out += ([path+'/'+k+': missing'] if k not in a or k not in b else differences(a[k],b[k],path+'/'+k,rtol,atol))
        return out
    if isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b): return [path+': length differs']
        return [d for i,(x,y) in enumerate(zip(a,b)) for d in differences(x,y,f'{path}/{i}',rtol,atol)]
    if type(a) is bool or type(b) is bool: return [] if type(a)==type(b) and a==b else [path+': boolean differs']
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        return [] if math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=rtol,abs_tol=atol) else [path+f': {a} != {b}']
    return [] if a==b else [path+': value differs']

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference',type=Path);p.add_argument('reproduced',type=Path)
    args=p.parse_args()
    assert not differences({'x':[1.,True]}, {'x':[1.+1e-13,True]})
    assert differences({'x':[1.]},{'x':[1.1]})
    assert differences({'x':1},{}) and differences(True,1)
    errors=differences(json.loads(args.reference.read_text(encoding='utf-8')),json.loads(args.reproduced.read_text(encoding='utf-8')))
    print(json.dumps({'mismatch_count':len(errors),'first_mismatches':errors[:20],'rtol':1e-10,'atol':1e-12},indent=2))
    raise SystemExit(bool(errors))
