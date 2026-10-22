"""Bundle local reviewed sources for manual Colab/Kaggle transfer; excludes raw data/weights."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('experiments/outputs/crossvul-source.tar.gz'))
    args=parser.parse_args();args.out.parent.mkdir(parents=True,exist_ok=True)
    files=[Path(x) for x in ('README.md','pyproject.toml','requirements.txt','requirements-dev.txt','requirements-cpu.lock','requirements-gpu.lock','LICENSE')]
    for folder in ('src','scripts','configs','tests','notebooks','docs/engineering','data/manifests'):
        files.extend(p for p in Path(folder).rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.md','.ipynb','.json') and '__pycache__' not in p.parts)
    with tarfile.open(args.out,'w:gz') as archive:
        for path in sorted(set(files)):archive.add(path,arcname=str(path),recursive=False)
    digest=hashlib.sha256(args.out.read_bytes()).hexdigest()
    report=dict(path=str(args.out),sha256=digest,bytes=args.out.stat().st_size,files=len(set(files)),
                exclusions='raw and processed datasets, pretrained weights, environments, checkpoints, results, caches, Git history')
    args.out.with_suffix('.manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
