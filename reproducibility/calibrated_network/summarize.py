from pathlib import Path
import json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
(ROOT/'generated').mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'reproducibility/calibrated_network'))
from run_operational_transfer import paired_boot
net=json.loads((ROOT/'reproducibility/calibrated_network/output/calibrated_network_corrected_20261029.json').read_text())
cal=json.loads((ROOT/'analysis/results/cp2_calibration_law_corrected_20261029.json').read_text())
stats={}
for law in net['laws']:
    rows=[r for r in net['records'] if r['law']==law]
    est={k:{'nmse':float(np.mean([r['estimators'][k]['mse']/r['estimators'][k]['energy'] for r in rows])),
            'se':float(np.mean([r['estimators'][k]['se_mean'] for r in rows]))} for k in net['estimators']}
    gains=np.array([r['estimators']['cal_B1']['se_mean']-r['estimators']['cal_B2']['se_mean'] for r in rows])
    prefixes={n:float(np.mean([r['estimators']['cal_B1']['uatf_prefix_per_user'][str(n)] for r in rows])) for n in (8,16,32,64)}
    stats[law]={'estimators':est,'gain':float(gains.mean()),'ci95':paired_boot(gains),'prefix_se':prefixes,
                'change_32_to_64':prefixes[64]-prefixes[32]}
(ROOT/'generated/network_summary.json').write_text(json.dumps(stats,indent=2))
print(json.dumps(stats,indent=2))
