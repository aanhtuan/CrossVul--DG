import json
import pytest
from crossvul_dg.preprocessing import (_parse_label,_parse_commit_date,_code_hash,_row_from_dict,iter_records)


def record(**kwargs):
    return dict(is_vul=True,func_before='int bad(){return 7;}',func='int fixed(){return 0;}',repo_name='owner/repo',commit_date=1577836800,**kwargs)

@pytest.mark.parametrize('value',[2,-1,.5,float('nan'),'yes','2',None,{},[]])
def test_invalid_labels(value):
    assert _parse_label(value) is None

@pytest.mark.parametrize('value',[True,1,'1','true',1.0])
def test_positive_labels(value):
    assert _parse_label(value)==1


def test_before_only_and_safe_mapping():
    row=record()
    assert _row_from_dict(row)['func']==row['func_before']
    row.pop('func_before')
    assert _row_from_dict(row) is None
    row['is_vul']=False
    assert _row_from_dict(row)['func']==row['func']
    row['label']=1
    assert _row_from_dict(row) is None

@pytest.mark.parametrize('stamp',[1577836800,1577836800000,1577836800000000,1577836800000000000,'2020-01-01T00:00:00Z'])
def test_date_units(stamp):
    assert str(_parse_commit_date(stamp))=='2020-01-01 00:00:00+00:00'

@pytest.mark.parametrize('stamp',[None,False,-1,0,'bad','3000-01-01',float('inf'),{}])
def test_invalid_dates(stamp):
    assert _parse_commit_date(stamp) is None


def test_hash_preserves_literals_and_owner_identity():
    assert _code_hash('int f() { /*x*/ return 1; }')==_code_hash('int f() { return 1; }')
    assert _code_hash('x="http://one";')!=_code_hash('x="http://two";')
    assert _code_hash('x="a b";')!=_code_hash('x="a  b";')
    assert _code_hash('int x;')!=_code_hash('intx;')
    assert _row_from_dict(record())['project']=='owner/repo'


def test_jsonl_invalid_is_not_silent(tmp_path):
    p=tmp_path/'raw.jsonl';p.write_text('{}\ninvalid\n')
    with pytest.raises(ValueError,match='malformed JSONL'):
        list(iter_records(p))

@pytest.mark.parametrize('kind',['json','jsonl'])
def test_stream_array_and_lines(tmp_path,kind):
    p=tmp_path/('raw.'+kind);row=record()
    p.write_text(json.dumps([row]) if kind=='json' else json.dumps(row)+'\n')
    assert list(iter_records(p))==[row]
