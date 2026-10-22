import json
import pandas as pd
import pytest
from crossvul_dg.preprocessing import prepare,sha256_file
from crossvul_dg.data import make_lopo_partitions,validate_lopo_frames


def fixture_rows(n_projects=4,n=40):
    rows=[]
    for p in range(n_projects):
        for i in range(n):
            rows.append(dict(is_vul=bool(i%2),func_before=f'int bad_{p}_{i}(){{return -1;}}',
                             func=f'int safe_{p}_{i}(){{return 0;}}',repo_name=f'owner/p{p}',
                             commit_date=1577836800+i*86400+p*3600,
                             cve_id=f'CVE-{p}-{i}',commit_hash=f'commit{p}_{i}'))
    return rows


def test_raw_parquet_lopo_and_idempotence(tmp_path):
    raw=tmp_path/'raw';raw.mkdir();out=tmp_path/'processed'
    (raw/'megavul.json').write_text(json.dumps(fixture_rows()))
    audit=prepare(raw,out,top_k=3,batch_size=7)
    assert audit['raw_records_parsed']==160
    assert audit['rows_after_validation_and_dedup']==160
    df=pd.read_parquet(out/'top10.parquet')
    tail=pd.read_parquet(out/'tail.parquet')
    assert len(df)==120 and len(tail)==40
    assert set(df['project']).isdisjoint(tail['project'])
    assert df[df.label==1].func.str.contains('bad_').all()
    frames,counts=make_lopo_partitions(df,'owner/p0')
    train,val,target,test=frames
    validate_lopo_frames(train,val,test,require_source_cells=True)
    assert target.commit_date.max()<test.commit_date.min()
    mtime=(out/'top10.parquet').stat().st_mtime_ns
    assert prepare(raw,out,top_k=3,batch_size=7)['signature']==audit['signature']
    assert (out/'top10.parquet').stat().st_mtime_ns==mtime
    assert sha256_file(out/'top10.parquet')==audit['output_hashes']['top10.parquet']


def test_quarantine_conflicts_cross_project_and_postfix(tmp_path):
    raw=tmp_path/'raw';raw.mkdir();rows=fixture_rows()
    conflict=rows[0].copy();conflict.update(is_vul=True,func_before=rows[0]['func'],func='fixed_conflict')
    cross=rows[2].copy();cross['repo_name']='other/project'
    postfix=rows[4].copy();postfix['func']=rows[1]['func']
    rows += [conflict,cross,postfix]
    (raw/'x.json').write_text(json.dumps(rows))
    audit=prepare(raw,tmp_path/'out')
    assert audit['quality']['conflicting_hashes']==1
    assert audit['quality']['cross_project_hashes']==1
    assert audit['quality']['postfix_overlap_rows']==1
    assert audit['quality']['quarantined_rows']==5


def test_failure_preserves_existing_artifacts(tmp_path):
    raw=tmp_path/'raw';raw.mkdir();out=tmp_path/'out';out.mkdir()
    old=out/'top10.parquet';old.write_bytes(b'original-user-data')
    (raw/'x.json').write_text('[{"is_vul":true}]')
    with pytest.raises(ValueError,match='No usable'):
        prepare(raw,out)
    assert old.read_bytes()==b'original-user-data'


def test_export_row_groups_remain_bounded(tmp_path):
    import pyarrow.parquet as pq
    raw=tmp_path/'raw';raw.mkdir()
    (raw/'x.json').write_text(json.dumps(fixture_rows(n_projects=2,n=1100)))
    prepare(raw,tmp_path/'out',top_k=2,batch_size=100)
    metadata=pq.ParquetFile(tmp_path/'out'/'top10.parquet').metadata
    assert metadata.num_row_groups>=2
    assert all(metadata.row_group(i).num_rows<=2048 for i in range(metadata.num_row_groups))
