"""Aggregate compatible completed experiments; fail closed on duplicates or gaps."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

METHOD_ORDER=['erm','da-coral','da-dann','irm','dro','cc-dro','pd-dro']
METHOD_LABEL={'erm':'ERM','irm':'IRM','dro':'GroupDRO','cc-dro':'CC-DRO',
              'pd-dro':'PD-DRO (+ labelled tail)','da-coral':'DA-CORAL (DA)','da-dann':'DA-DANN (DA)'}
METRICS=['pr_auc','roc_auc','f1','precision','recall','recall_at_p50','recall_at_validation_p50_threshold']
REQUIRED_METRICS=['pr_auc','roc_auc','f1','precision','recall','recall_at_p50']


def load_runs(runs_dir,backbone_filter=None):
    """cohort -> method -> held-out -> seed -> test metrics, with provenance."""
    bucket=defaultdict(lambda:defaultdict(lambda:defaultdict(dict)))
    metadata={}; method_signatures={}
    for path in sorted(Path(runs_dir).rglob('results.json')):
        result=json.loads(path.read_text())
        if result.get('status')!='completed' or result.get('result_version')!=2:
            raise ValueError(f'Unverified, legacy or incomplete result: {path}')
        args=result['args']; cfg=args.get('resolved_config')
        if not isinstance(cfg,dict) or not result.get('comparison_signature') or not result.get('experiment_signature'):
            raise ValueError(f'Missing resolved experiment provenance: {path}')
        backbone=cfg['encoder']['name']
        if backbone_filter and backbone.split('/')[-1] not in backbone_filter:
            continue
        source=result.get('software',{}).get('source_signature')
        if not source:
            raise ValueError(f'Missing executable-source signature: {path}')
        cohort=f"{backbone}@{result['comparison_signature']}@{source}"
        key=(cohort,args['method'])
        method_identity=json.dumps({'config':cfg,'tail':result.get('tail_data_provenance')},sort_keys=True)
        # Runtime-specific checkpoint/device fields are irrelevant to compatibility.
        normalized=json.loads(method_identity)
        for field in ('checkpoint_path','resume','device','num_workers'):
            normalized['config'].get('train',{}).pop(field,None)
        method_identity=json.dumps(normalized,sort_keys=True)
        if key in method_signatures and method_signatures[key]!=method_identity:
            raise ValueError(f'Incompatible runs within method/cohort: {path}')
        method_signatures[key]=method_identity
        fold=bucket[cohort][args['method']][args['held_out']]
        seed=int(args['seed'])
        if seed in fold:
            raise ValueError(f'Duplicate fold/seed result: {path}')
        metrics=result['test']
        for metric in REQUIRED_METRICS:
            if not isinstance(metrics.get(metric),(int,float)) or not np.isfinite(metrics[metric]):
                raise ValueError(f'Undefined or missing metric {metric}: {path}')
        fold[seed]=metrics
        metadata[cohort]=dict(backbone=backbone,comparison_signature=result['comparison_signature'],source_signature=source,
                              config=normalized['config'],data=result.get('data_provenance'))
    if not bucket:
        raise ValueError('No matching completed results.json files found')
    return bucket,metadata


def fold_means(folds,metric):
    out={}
    for held,seeds in folds.items():
        values=[m[metric] for m in seeds.values() if isinstance(m.get(metric),(int,float)) and np.isfinite(m[metric])]
        if values: out[held]=float(np.mean(values))
    return out


def summarize(bucket,expected_folds=10,expected_seeds=3,allow_incomplete=False,expected_methods=None):
    summaries={}
    for cohort,methods in bucket.items():
        issues=[]
        if expected_methods and set(methods)!=set(expected_methods):
            issues.append(f'methods {sorted(methods)} differ from expected {sorted(expected_methods)}')
        union_folds=set().union(*(set(folds) for folds in methods.values()))
        union_seeds=set().union(*(set(seeds) for folds in methods.values() for seeds in folds.values()))
        for method,folds in methods.items():
            if len(folds)!=expected_folds or set(folds)!=union_folds:
                issues.append(f'{method}: incomplete/mismatched folds {sorted(folds)}')
            for held,seeds in folds.items():
                if len(seeds)!=expected_seeds or set(seeds)!=union_seeds:
                    issues.append(f'{method}/{held}: incomplete/mismatched seeds {sorted(seeds)}')
        if issues and not allow_incomplete:
            raise ValueError('Incomplete experiments: '+'; '.join(issues))
        rows={}
        for method,folds in methods.items():
            row={}
            for metric in METRICS:
                fm=fold_means(folds,metric)
                if fm:
                    values=list(fm.values())
                    row[metric]=dict(mean=float(np.mean(values)),std=float(np.std(values)),worst=float(min(values)),per_fold=fm,
                                     n_folds=len(fm),n_runs=sum(metric in m for seeds in folds.values() for m in seeds.values()))
            rows[method]=row
        comparisons=[]
        # Only complete, exactly matched folds/seeds support inferential comparisons.
        if not issues:
            for method,baseline in [(m,'erm') for m in methods if m!='erm']+[('cc-dro','dro')]:
                if method not in methods or baseline not in methods: continue
                a=fold_means(methods[method],'pr_auc');b=fold_means(methods[baseline],'pr_auc')
                names=sorted(a)
                if len(names)>=6:
                    delta=np.array([a[p]-b[p] for p in names])
                    pvalue=1. if np.allclose(delta,0) else float(wilcoxon(delta).pvalue)
                    comparisons.append(dict(method=method,baseline=baseline,n_folds=len(names),p_value_unadjusted=pvalue,
                                            note='exploratory; multiple comparisons unadjusted'))
        summaries[cohort]=dict(status='exploratory_incomplete' if issues else 'complete',issues=issues,
                               methods=rows,comparisons=comparisons,
                               metric_note='pr_auc=Average Precision; f1=binary vulnerable-class F1 averaged over projects; recall_at_p50=PR-curve diagnostic, not deployed threshold')
    return summaries


def write_latex(summary,path):
    lines=['% Generated from verified experiments; cohort-specific.',
           '% DA uses unlabelled target; PD-DRO uses extra labelled tail.',
           '\\begin{tabular}{lrrrr}','Method & AP & F1 & ROC-AUC & Worst F1 \\\\','\\hline']
    for method in METHOD_ORDER:
        if method not in summary['methods']: continue
        row=summary['methods'][method]
        vals=[row[m]['mean']*100 for m in ('pr_auc','f1','roc_auc')]+[row['f1']['worst']*100]
        lines.append(METHOD_LABEL[method]+' & '+' & '.join(f'{x:.2f}' for x in vals)+' \\\\')
    lines+=['\\end{tabular}','']
    Path(path).write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs_dir',type=Path,required=True);p.add_argument('--out',type=Path,default=Path('paper_tables'))
    p.add_argument('--expected_folds',type=int,default=10);p.add_argument('--expected_seeds',type=int,default=3)
    p.add_argument('--expected_methods',nargs='+',choices=METHOD_ORDER)
    p.add_argument('--allow_incomplete',action='store_true');p.add_argument('--backbone_filter',nargs='*')
    args=p.parse_args()
    bucket,metadata=load_runs(args.runs_dir,args.backbone_filter)
    summary=summarize(bucket,args.expected_folds,args.expected_seeds,args.allow_incomplete,args.expected_methods)
    args.out.mkdir(parents=True,exist_ok=True)
    for i,(cohort,value) in enumerate(summary.items()):
        value['provenance']=metadata[cohort]
        write_latex(value,args.out/f'cohort_{i}_table.tex')
        print(json.dumps(dict(cohort=cohort[:90],status=value['status'],methods=list(value['methods']),issues=value['issues'])))
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))

if __name__=='__main__': main()
