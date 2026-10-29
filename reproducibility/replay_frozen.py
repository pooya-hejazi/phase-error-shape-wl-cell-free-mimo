"""Portable replay of an unchanged frozen experiment in a new generated directory.

The historical machine/path lock is replaced by explicit package/version checks;
the frozen driver, configuration, seeds, checksums and numerical code are unchanged.
"""
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKETS = {
    'v08': ('v08_run', 'docs/claude_review/R2C01_DIRA_SHAPE_v08'),
    'replication': ('system_run', 'docs/claude_review/R2C01_DIRA_SHAPE_REP_v01'),
    'expanded': ('expanded_run_v02', 'reproducibility/expanded_run_v02'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('packet', choices=PACKETS)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run-name', help='New directory name below generated/')
    args = parser.parse_args()
    packet, relative = PACKETS[args.packet]
    name=args.run_name or 'replay_'+args.packet
    if Path(name).name != name or name in ('.','..'):
        parser.error('--run-name must be one directory name')
    base = ROOT/'generated'/name
    base.mkdir(parents=True, exist_ok=False)
    source = ROOT/'reproducibility'/packet
    work = base/relative
    run = work/'run'
    shutil.copytree(source if args.packet == 'replication' else source/'run', run,
                    ignore=shutil.ignore_patterns('__pycache__', 'output', 'temp', '*.started'))
    shutil.copy2(source/'ARTIFACT_MANIFEST.json', work/'ARTIFACT_MANIFEST.json')
    shutil.copytree(ROOT/'reproducibility/v08_run/vendor/phase_graph_cme', base/'src/phase_graph_cme')
    # The expanded packet imports its own subset first; supply its absent modules.
    if args.packet == 'expanded':
        for file in (base/'src/phase_graph_cme').glob('*.py'):
            target=run/'phase_graph_cme'/file.name
            if not target.exists(): shutil.copy2(file,target)
    (run/'temp').mkdir(exist_ok=True)
    os.environ.update({name:'1' for name in ('PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
    os.environ.update(TEMP=str(run/'temp'),TMP=str(run/'temp'))
    spec=importlib.util.spec_from_file_location('frozen_driver',run/'E10_shape_run.py')
    driver=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    assert driver.ROOT.resolve()==base.resolve(), (driver.ROOT,base)
    (base/driver.OUTPUT_REL).parent.mkdir(parents=True,exist_ok=True)

    def portable_environment():
        if sys.version_info[:2]!=(3,12) or not sys.flags.dont_write_bytecode:
            raise RuntimeError('Use Python 3.12 with -B')
        versions={k:importlib.metadata.version(k) for k in driver.PINNED}
        if versions != driver.PINNED:
            raise RuntimeError(f'Install requirements.txt; observed {versions}')
        return {'python':sys.version,'platform':platform.platform(),'pinned_versions':versions,
                'mode':'portable replay; historical machine/path lock replaced; numerical inputs unchanged'}
    driver.environment_checks=portable_environment
    environment=portable_environment()
    # Git may convert text newlines. Restore only bytes matching the frozen hash.
    def restore_newlines(path, expected):
        raw=path.read_bytes()
        lf=raw.replace(b'\r\n',b'\n')
        for candidate in (raw,lf,lf.replace(b'\n',b'\r\n')):
            if hashlib.sha256(candidate).hexdigest()==expected:
                path.write_bytes(candidate)
                return
        raise RuntimeError(f'Frozen content mismatch (not newlines): {path}')
    manifest=json.loads((base/driver.MANIFEST_REL).read_text(encoding='utf-8'))
    listing=base/driver.CHECKSUMS_REL
    restore_newlines(listing,manifest['inputs_checksum']['sha256'])
    for line in listing.read_text().splitlines():
        digest,relative_file=line.split(maxsplit=1)
        restore_newlines(base/relative_file.strip(),digest)
    driver.verify_inputs()
    driver.load_config()
    (base/'PORTABLE_ENVIRONMENT.json').write_text(json.dumps(environment,indent=2))
    if args.prepare_only:
        print('PASS: frozen checksums, configuration and portable environment:',base)
        return
    sys.argv=[str(run/'E10_shape_run.py'),'--gate-confirmation',driver.GATE_TOKEN,'--output',driver.OUTPUT_REL]
    code=driver.main()
    if code: raise SystemExit(code)
    sys.path.insert(0,str(ROOT/'reproducibility/v08_run'))
    from replay import compare,scientific_view
    stored=source/('run/output/E10_shape_output.json' if args.packet=='v08' else 'output/E10_shape_output.json')
    fresh=base/driver.OUTPUT_REL
    differences,max_diff=compare(scientific_view(json.loads(stored.read_text(encoding='utf-8'))),
                                 scientific_view(json.loads(fresh.read_text(encoding='utf-8'))))
    report={'packet':args.packet,'mismatch_count':len(differences),'mismatches':differences[:50],
            'max_relative_difference':max_diff,'tolerance':1e-12}
    (base/'COMPARISON.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if differences: raise SystemExit(1)


if __name__=='__main__': main()
