import copy
import json
from pathlib import Path
import sys
import pytest
import yaml
import run_lopo
import aggregate_results
from aggregate_results import load_runs,summarize,METHOD_ORDER
from crossvul_dg.preprocessing import prepare
from crossvul_dg.provenance import ensure_result_identity,verify_processed
from conftest import TinyModel,Tokenizer


def real_schema_fixture():
    # Purely synthetic test rows, never written to repository data/.
    rows=[]
    for p in range(4):
        for i in range(40):
            rows.append(dict(is_vul=bool(i%2),func_before=f'bad_{p}_{i}()',func=f'good_{p}_{i}()',
                             repo_name=f'p{p}',commit_date=1577836800+i*86400+p*3600,
                             cve_id=f'CVE-{p}-{i}',commit_hash=f'{p}_{i}'))
    return rows

@pytest.mark.parametrize('method',METHOD_ORDER)
def test_actual_runner_with_mock_backbone_to_verified_summary(tmp_path,monkeypatch,method):
    raw=tmp_path/'raw';raw.mkdir();(raw/'x.json').write_text(json.dumps(real_schema_fixture()))
    processed=tmp_path/'processed';prepare(raw,processed,top_k=3)
    config=yaml.safe_load(Path('configs/default.yaml').read_text())
    config['train'].update(epochs=1,batch_size=4,warmup_steps=0,lr=.01)
    config['encoder']['max_length']=12;config['eval']['batch_size']=4
    config['pseudo'].update(n_clusters=1,embedding_subset=4)
    config_path=tmp_path/'config.yaml';config_path.write_text(yaml.safe_dump(config))
    output=tmp_path/'runs'/'one'
    argv=['run_lopo','--config',str(config_path),'--method',method,'--held_out','p0',
          '--data_path',str(processed/'top10.parquet'),'--tail_path',str(processed/'tail.parquet'),
          '--output_dir',str(output)]
    monkeypatch.setattr(sys,'argv',argv)
    monkeypatch.setattr(run_lopo.AutoTokenizer,'from_pretrained',lambda *a,**k:Tokenizer())
    monkeypatch.setattr(run_lopo,'EncoderClassifier',lambda **k:TinyModel())
    monkeypatch.setattr(run_lopo,'require_headroom',lambda: {'test_fixture':True})
    run_lopo.main()
    result=json.loads((output/'results.json').read_text())
    assert result['status']=='completed'
    assert result['split_audit']['partition_counts']['test']['n']==20
    assert result['args']['resolved_config']['pseudo']['n_clusters']==1
    assert (output/'latest.pt').exists() and (output/'latest.best.pt').exists()
    assert len(result['sampling_budget'])==1 if method in ('erm','da-coral','da-dann') else len(result['sampling_budget'])>=2
    if method=='pd-dro':
        assert result['auxiliary_data']['additional_labeled_rows']>0
        assert result['regime']=='DG+additional-labelled-tail'
    bucket,meta=load_runs(tmp_path/'runs')
    summary=summarize(bucket,1,1,expected_methods=[method])
    assert next(iter(summary.values()))['status']=='complete'
    monkeypatch.setattr(sys,'argv',['aggregate_results','--runs_dir',str(tmp_path/'runs'),'--out',str(tmp_path/'tables'),
                                  '--expected_folds','1','--expected_seeds','1','--expected_methods',method])
    aggregate_results.main()
    assert json.loads((tmp_path/'tables'/'summary.json').read_text())
    assert (tmp_path/'tables'/'cohort_0_table.tex').exists()
    monkeypatch.setattr(sys,'argv',argv)
    # Re-run skips only after verifying EXACT experiment identity.
    monkeypatch.setattr(run_lopo,'EncoderClassifier',lambda **k:pytest.fail('matching completed run must not initialize model'))
    run_lopo.main()
    (output/'results.json').write_text(json.dumps(result|{'experiment_signature':'stale'}))
    with pytest.raises(ValueError,match='identity mismatch'): run_lopo.main()


def test_aggregation_duplicates_incomplete_and_cohorts(tmp_path):
    cfg=yaml.safe_load(Path('configs/default.yaml').read_text())
    def item(comparison='same',method='cc-dro',seed=0,config=None):
        return dict(status='completed',result_version=2,comparison_signature=comparison,experiment_signature='unique',
                    software={'source_signature':'abc'},args=dict(resolved_config=config or cfg,method=method,held_out='a',seed=seed),
                    test={metric:.5 for metric in ['pr_auc','roc_auc','f1','precision','recall','recall_at_p50']})
    def write(name,obj):
        path=tmp_path/name;path.mkdir();(path/'results.json').write_text(json.dumps(obj))
    write('a',item())
    bucket,_=load_runs(tmp_path)
    with pytest.raises(ValueError,match='Incomplete'): summarize(bucket,10,3)
    assert next(iter(summarize(bucket,10,3,True).values()))['status']=='exploratory_incomplete'
    with pytest.raises(ValueError,match='methods'): summarize(bucket,1,1,expected_methods=['erm','cc-dro'])
    write('b',item(comparison='different'))
    assert len(load_runs(tmp_path)[0])==2
    write('duplicate',item())
    with pytest.raises(ValueError,match='Duplicate'): load_runs(tmp_path)


def test_legacy_results_and_tampered_data_are_rejected(tmp_path):
    (tmp_path/'results.json').write_text(json.dumps({'args':{},'test':{}}))
    with pytest.raises(ValueError,match='legacy'): load_runs(tmp_path)
    raw=tmp_path/'raw';raw.mkdir();(raw/'x.json').write_text(json.dumps(real_schema_fixture()))
    out=tmp_path/'processed';prepare(raw,out)
    (out/'top10.parquet').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='hash mismatch'): verify_processed(out/'top10.parquet')


def test_notebook_includes_ccdro_and_parses_resolved_config():
    notebook=json.loads(Path('notebooks/analysis.ipynb').read_text())
    text=''.join(''.join(c['source']) for c in notebook['cells'])
    assert 'METHOD_ORDER' in text and 'cc-dro' in METHOD_ORDER
    assert 'resolved_config' in text
    assert not any(c.get('outputs') for c in notebook['cells'])
    for cell in notebook['cells']:
        if cell['cell_type']=='code': compile(''.join(cell['source']),'analysis.ipynb','exec')


def test_notebook_renders_ccdro_chart_from_verified_summary(tmp_path,monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    notebook=json.loads(Path('notebooks/analysis.ipynb').read_text())
    cohort='synthetic-verification-only'
    namespace=dict(summary={cohort:{'status':'complete','methods':{'cc-dro':{'pr_auc':{'mean':.5,'std':.1}}}}},
                   metadata={cohort:{'backbone':'synthetic-test-only'}},METHOD_ORDER=METHOD_ORDER,
                   METHOD_LABEL=aggregate_results.METHOD_LABEL,ROOT=tmp_path,plt=plt)
    labels=[]
    monkeypatch.setattr(plt,'show',lambda:labels.extend(t.get_text() for t in plt.gca().get_xticklabels()))
    plot=next(c for c in notebook['cells'] if c['cell_type']=='code' and 'ax.bar' in ''.join(c['source']))
    exec(''.join(plot['source']),namespace)
    assert 'CC-DRO' in labels
    assert (tmp_path/'figures'/'cohort_0_ap.png').stat().st_size>0
    plt.close('all')


def test_validation_only_never_forwards_test_then_evaluates_same_checkpoint(tmp_path,monkeypatch):
    raw=tmp_path/'raw';raw.mkdir();(raw/'x.json').write_text(json.dumps(real_schema_fixture()))
    processed=tmp_path/'processed';prepare(raw,processed,top_k=3)
    cfg=yaml.safe_load(Path('configs/default.yaml').read_text());cfg['train'].update(epochs=1,warmup_steps=0)
    cfg['encoder']['max_length']=12
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump(cfg));out=tmp_path/'run'
    args=['run_lopo','--config',str(config),'--method','cc-dro','--held_out','p0','--data_path',str(processed/'top10.parquet'),'--output_dir',str(out)]
    monkeypatch.setattr(run_lopo,'require_headroom',lambda:{'fixture':True})
    monkeypatch.setattr(run_lopo.AutoTokenizer,'from_pretrained',lambda *a,**kw:Tokenizer())
    monkeypatch.setattr(run_lopo,'EncoderClassifier',lambda **kw:TinyModel())
    original_builder=run_lopo.build_trainer
    def guarded(*a,**kw):
        trainer=original_builder(*a,**kw);predict=trainer.predict
        def no_test(loader):
            assert 'p0' not in set(loader.dataset.df.project),'selection must never forward held-out test'
            return predict(loader)
        trainer.predict=no_test
        return trainer
    monkeypatch.setattr(run_lopo,'build_trainer',guarded)
    monkeypatch.setattr(sys,'argv',args+['--selection_only']);run_lopo.main()
    selection=json.loads((out/'selection.json').read_text())
    assert selection['status']=='validation_complete' and 'test' not in selection
    assert not (out/'results.json').exists()
    monkeypatch.setattr(run_lopo,'build_trainer',original_builder)
    monkeypatch.setattr(sys,'argv',args);run_lopo.main()
    completed=json.loads((out/'results.json').read_text())
    assert completed['experiment_signature']==selection['experiment_signature']
    assert len(completed['training_history'])==len(selection['training_history'])==1
