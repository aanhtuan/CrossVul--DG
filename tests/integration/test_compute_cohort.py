import json
import pandas as pd
from crossvul_dg.cohorts import bounded_frames,materialize
from crossvul_dg.data import load_processed
from crossvul_dg.preprocessing import prepare
from test_preprocess import fixture_rows


def test_metadata_loading_and_selected_materialization(tmp_path):
    raw=tmp_path/'raw';raw.mkdir();(raw/'x.json').write_text(json.dumps(fixture_rows()))
    out=tmp_path/'processed';prepare(raw,out)
    metadata=load_processed(out/'top10.parquet',metadata_only=True)
    assert 'func' not in metadata
    chosen=metadata.head(5)
    result=materialize(out/'top10.parquet',chosen)
    assert result.code_hash.tolist()==chosen.code_hash.tolist()
    assert len(result)==5 and result.func.str.len().min()>0


def test_compute_budget_is_explicit_and_eval_hashes_ignore_labels():
    from conftest import frame
    def data(project,start):
        result=frame(project,start,20);result['code_hash']=[project+str(i) for i in range(20)];return result
    frames=[pd.concat([data('a',0),data('b',0)],ignore_index=True),data('a',100),data('held',200),data('held',300)]
    result,audit=bounded_frames(frames,train_per_project=6,val_limit=4,test_limit=5,seed=7)
    assert [len(x) for x in result]==[12,4,20,5]
    assert all(set(group.label)=={0,1} for _,group in result[0].groupby('project'))
    assert audit['protocol']=='bounded-real-lopo-v1'
    changed=[frame.copy() for frame in frames]
    changed[1]['label']=1-changed[1].label;changed[3]['label']=1-changed[3].label
    again,_=bounded_frames(changed,train_per_project=6,val_limit=4,test_limit=5,seed=7)
    assert again[1].code_hash.tolist()==result[1].code_hash.tolist()
    assert again[3].code_hash.tolist()==result[3].code_hash.tolist()
