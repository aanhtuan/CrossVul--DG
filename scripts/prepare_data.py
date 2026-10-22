"""Strict streaming MegaVul preprocessing, protocol v2. See engineering report."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')

import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from crossvul_dg.preprocessing import prepare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw_dir', type=Path, default=Path('data/raw/megavul'))
    p.add_argument('--out_dir', type=Path, default=Path('data/processed'))
    p.add_argument('--top_k', type=int, default=10)
    for name, value in [('train_ratio',.8),('val_ratio',.1),('test_ratio',.1)]:
        p.add_argument('--'+name, type=float, default=value)
    p.add_argument('--batch_size', type=int, default=1000)
    p.add_argument('--memory_limit', default='256MB')
    p.add_argument('--as_of', help='Reproducible upper date bound; UTC ISO timestamp recorded in manifest')
    p.add_argument('--dedup', action='store_true', help='Compatibility flag; strict dedup is always enforced in v2')
    args = vars(p.parse_args())
    args.pop('dedup')
    result = prepare(**args)
    print(json.dumps({k:result[k] for k in ('signature','raw_records_parsed','exclusions','rows_after_validation_and_dedup','top_project_counts')}, indent=2))

if __name__ == '__main__':
    main()
