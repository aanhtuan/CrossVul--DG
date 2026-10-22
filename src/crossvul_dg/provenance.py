"""Stable experiment identities and data/software provenance."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from .preprocessing import PROTOCOL,sha256_file


def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str,separators=(',',':')).encode()).hexdigest()


def verify_processed(path):
    path=Path(path)
    marker=path.parent/'splits.json'
    if not marker.exists():
        raise ValueError('Processed data missing splits.json commit marker; reprocess using protocol v2')
    manifest=json.loads(marker.read_text())
    if manifest.get('protocol')!=PROTOCOL:
        raise ValueError('Legacy/incompatible processed-data protocol; reprocess raw MegaVul')
    expected=manifest.get('output_hashes',{}).get(path.name)
    actual=sha256_file(path)
    if not expected or expected!=actual:
        raise ValueError(f'Processed data hash mismatch: {path.name}')
    return dict(sha256=actual,manifest_signature=manifest['signature'],protocol=PROTOCOL,
                input_hashes=manifest['inputs'])


def experiment_identity(cfg,method,held_out,seed,data,tail=None):
    # Paths, resume preference and hardware location do not define the objective.
    effective=json.loads(json.dumps(cfg))
    for key in ('checkpoint_path','resume','device','num_workers'):
        effective.get('train',{}).pop(key,None)
    comparison=dict(version='experiment-v2',config=effective,data=data)
    comparison_signature=signature(comparison)
    experiment=dict(comparison=comparison_signature,method=method,held_out=held_out,seed=seed,tail=tail)
    return signature(experiment),comparison_signature


def software_provenance():
    packages={}
    for name in ('torch','transformers','pandas','numpy','pyarrow','scikit-learn','duckdb','ijson'):
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]=None
    def git(*args):
        try: return subprocess.check_output(['git',*args],stderr=subprocess.DEVNULL,text=True).strip()
        except (OSError,subprocess.CalledProcessError): return None
    # Hash the executable source as well as recording a possibly dirty Git commit.
    root=Path(__file__).resolve().parents[2]
    source_hashes={}
    for directory in ('src','scripts','configs'):
        for file in sorted((root/directory).rglob('*')):
            if file.is_file() and file.suffix in ('.py','.yaml','.sh'):
                source_hashes[str(file.relative_to(root))]=sha256_file(file)
    return dict(python=platform.python_version(),platform=platform.platform(),packages=packages,
                git_commit=git('rev-parse','HEAD'),git_dirty=bool(git('status','--porcelain')),
                source_signature=signature(source_hashes),source_hashes=source_hashes)


def ensure_result_identity(path,identity):
    path=Path(path)
    if not path.exists(): return False
    data=json.loads(path.read_text())
    if data.get('status')!='completed' or data.get('experiment_signature')!=identity:
        raise ValueError('Existing result experiment identity mismatch or incomplete result; use a new output directory')
    return True
