"""Verify release hashes, exclusion rules and paper-matched experimental sizes."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
for name,digest in manifest['files'].items():
    p=ROOT/name
    assert p.resolve().is_relative_to(ROOT.resolve()),name
    assert p.suffix.lower() not in {'.pdf','.tex','.bib','.docx','.zip'},name
    assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
def load(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
net=load('reproducibility/calibrated_network/output/calibrated_network_corrected_20261029.json')
assert (net['topologies'],net['realizations'],len(net['records']))==(32,64,128)
transfer=load('reproducibility/calibrated_network/output/operational_transfer_corrected_20261029.json')
assert (transfer['topologies'],transfer['realizations'])==(8,64)
dl=load('reproducibility/downlink/downlink_convergence.json')
assert (dl['topologies'],dl['frames'])==(32,128)
cal=load('analysis/results/cp2_calibration_law_corrected_20261029.json')
expected=[.9838912774,.9656040420,.9608911349,.9575400558]
for law,value in zip(net['laws'],expected):
    row=next(r for r in cal['laws'][law]['curve'] if r['frames']==1024)
    assert abs(row['capture_mean']-value)<1e-9,(law,row)
print(json.dumps({'status':'PASS','version':manifest['version'],'files_verified':len(manifest['files']),
                  'paper_experimental_sizes_verified':True,'prohibited_release_files':0},indent=2))
