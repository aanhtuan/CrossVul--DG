import copy
import json
from pathlib import Path
import pandas as pd
import pytest
import yaml
from crossvul_dg.data import temporal_split,split_held_out,purge_group_overlap,assert_no_leakage,ClassBalancedBatchSampler
from crossvul_dg.config import validate
from crossvul_dg.utils import apply_overrides,save_json
from conftest import frame


def test_equal_timestamp_never_crosses_boundary():
    df=frame('a',n=10)
    df.loc[7:9,'commit_date']=df.loc[7,'commit_date']
    train,val,test=temporal_split(df,.7,.1,.2)
    assert set(train.commit_date).isdisjoint(val.commit_date)
    assert set(val.commit_date).isdisjoint(test.commit_date)


def test_group_metadata_purges_all_members_and_aliases():
    a,b=frame('a'),frame('b',20)
    a['cve_id']='';b['cve_id']=''
    a.loc[0,'cve_id']='shared'
    b.loc[0,'provenance']=json.dumps({'cve_id':['shared'],'commit_hash':[]})
    with pytest.raises(ValueError,match='leakage'): assert_no_leakage([a,b])
    x,y=purge_group_overlap([a,b])
    assert len(x)==len(a)-1 and len(y)==len(b)-1
    assert_no_leakage([x,y])


def test_ratios_and_balanced_sampler_are_checked():
    with pytest.raises(ValueError): split_held_out(frame('a'),1.)
    with pytest.raises(ValueError): temporal_split(frame('a'),.8,-.1,.3)
    with pytest.raises(ValueError): ClassBalancedBatchSampler([0,0,0],4)


def test_configs_and_strict_overrides():
    for path in Path('configs').glob('*.yaml'):
        assert validate(yaml.safe_load(path.read_text()))==[]
    cfg=yaml.safe_load(Path('configs/default.yaml').read_text())
    assert apply_overrides(copy.deepcopy(cfg),['train.lr=0.0001'])['train']['lr']==.0001
    with pytest.raises(ValueError,match='Unknown config key'): apply_overrides(cfg,['train.lrr=1'])
    broken=copy.deepcopy(cfg);broken['train']['lr']=0
    assert any('train.lr' in e for e in validate(broken))
    broken=copy.deepcopy(cfg);broken['encoder']['revision']='main'
    assert any('immutable' in e for e in validate(broken))
    broken=copy.deepcopy(cfg);broken['cc_dro']['lambda_inv']=-1
    assert validate(broken)


def test_results_json_is_atomic_and_standard(tmp_path):
    path=tmp_path/'results.json';save_json({'undefined':float('nan')},path)
    assert json.loads(path.read_text())=={'undefined':None}
    assert not list(tmp_path.glob('*.tmp'))


def test_resource_guard_rejects_insufficient_ram(monkeypatch):
    import crossvul_dg.resources as resources
    monkeypatch.setattr(resources,'snapshot',lambda:dict(available_ram_bytes=2*1024**3,disk_free_bytes=100*1024**3))
    with pytest.raises(RuntimeError,match='Insufficient resources'): resources.require_headroom()


def test_near_duplicate_screen_is_bounded_and_marked_diagnostic(tmp_path):
    from audit_data import near_duplicate_candidates
    df=frame('a',n=10)
    df['code_hash']=[str(i) for i in range(10)]
    path=tmp_path/'x.parquet';df.to_parquet(path)
    result=near_duplicate_candidates(path,max_records=3,max_comparisons=2)
    assert result['examined_rows']<=3 and result['comparisons']<=2
    assert result['capped'] and 'not an exhaustive' in result['note']
