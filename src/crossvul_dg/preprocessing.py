"""Streaming MegaVul ingestion. Protocol v2 intentionally changes legacy populations."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROTOCOL = 'megavul-v2-strict-before-global-quarantine-temporal-purge'
SOURCE = 'https://github.com/Icyrockton/MegaVul'
SPEC = SOURCE + '/blob/main/SPECIFICATION.md'


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _parse_label(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    if isinstance(value, str) and value.strip().lower() in ('0', '1', 'true', 'false'):
        return int(value.strip().lower() in ('1', 'true'))
    return None


def _parse_commit_date(value, max_date=None):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            return None
        unit = 'ns' if number > 1e17 else 'us' if number > 1e14 else 'ms' if number > 1e11 else 's'
        result = pd.to_datetime(number, unit=unit, utc=True, errors='coerce')
    except (TypeError, ValueError, OverflowError):
        if not isinstance(value, str) or not value.strip():
            return None
        result = pd.to_datetime(value, utc=True, errors='coerce')
    if pd.isna(result) or result.year < 1970 or result > (pd.Timestamp(max_date) if max_date is not None else pd.Timestamp.now(tz='UTC')):
        return None
    return result


# Lex C/C++ strings before comments: URLs and spaces inside literals are significant.
_LEX = re.compile(r'R"([^ ()\\\t\r\n]{0,16})\(.*?\)\1"|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*|\s+|.', re.DOTALL)


def _normalize_code(code):
    tokens = []
    for match in _LEX.finditer(code):
        token = match.group()
        if token.isspace() or token.startswith(('/*', '//')):
            # Preserve a boundary so `int x` cannot collide with `intx`.
            if tokens and tokens[-1] != ' ':
                tokens.append(' ')
        else:
            tokens.append(token)
    return ''.join(tokens).strip()


def _code_hash(code):
    return hashlib.sha256(_normalize_code(code).encode('utf-8')).hexdigest()


def parse_record(record, max_date=None):
    if not isinstance(record, dict):
        return None, 'non_object'
    labels = [_parse_label(record[k]) for k in ('is_vul', 'label') if k in record]
    if not labels or None in labels:
        return None, 'invalid_label'
    if len(set(labels)) != 1:
        return None, 'conflicting_label_fields'
    label = labels[0]
    field = 'func_before' if label else 'func'
    code = record.get(field)
    if not isinstance(code, str) or not code.strip():
        return None, 'missing_vulnerable_before' if label else 'missing_safe_func'
    project = record.get('repo_name', record.get('project'))
    if not isinstance(project, str) or not project.strip():
        return None, 'missing_project'
    # Keep owner/repository names intact: repo basenames are not unique domains.
    date = _parse_commit_date(record.get('commit_date'),max_date)
    if date is None:
        return None, 'invalid_or_missing_date'
    for key in ('cve_id', 'commit_hash'):
        if record.get(key) is not None and not isinstance(record[key], str):
            return None, 'invalid_' + key
    if record.get('cwe_ids') is not None and (not isinstance(record['cwe_ids'], list) or
            not all(isinstance(item, str) for item in record['cwe_ids'])):
        return None, 'invalid_cwe_ids'
    if record.get('diff_func') is not None and not isinstance(record['diff_func'], str):
        return None, 'invalid_diff_func'
    code = code.strip()
    post = record.get('func') if label else None
    if label and (not isinstance(post, str) or not post.strip()):
        return None, 'missing_postfix_for_audit'
    if label and _code_hash(code) == _code_hash(post):
        return None, 'unchanged_before_after'
    return dict(func=code, label=label, project=project.strip().removesuffix('.git').rstrip('/'),
                commit_date=date, cve_id=record.get('cve_id') or '',
                commit_hash=record.get('commit_hash') or '', code_hash=_code_hash(code),
                post_hash=_code_hash(post) if label else '', code_chars=len(code)), None


def _row_from_dict(record):
    return parse_record(record)[0]


def iter_records(path):
    import ijson
    with Path(path).open('rb') as f:
        first = next((ch for ch in iter(lambda: f.read(1), b'') if not ch.isspace()), b'')
        f.seek(0)
        if first == b'[':
            yield from ijson.items(f, 'item', use_float=True)
        else:
            for line_no, line in enumerate(f, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except (ValueError, UnicodeDecodeError) as exc:
                        raise ValueError(f'{path}:{line_no}: malformed JSONL') from exc


def prepare(raw_dir, out_dir, *, top_k=10, train_ratio=.8, val_ratio=.1,
            test_ratio=.1, batch_size=1000, memory_limit='256MB', manifests_dir=None, as_of=None):
    import duckdb
    import os
    import tempfile
    import shutil
    raw_dir, out_dir = Path(raw_dir).resolve(), Path(out_dir).resolve()
    if raw_dir == out_dir or raw_dir in out_dir.parents or out_dir in raw_dir.parents:
        raise ValueError('Raw and processed directories must be separate siblings')
    if top_k < 1 or batch_size < 1 or min(train_ratio, val_ratio, test_ratio) <= 0 or abs(train_ratio + val_ratio + test_ratio - 1) > 1e-8:
        raise ValueError('Invalid preprocessing parameters')
    files = sorted(p for p in raw_dir.rglob('*') if p.suffix in ('.json', '.jsonl') and p.is_file())
    if not files:
        raise FileNotFoundError(f'No JSON/JSONL dataset found under {raw_dir}')
    inputs = [dict(name=str(p.relative_to(raw_dir)), size_bytes=p.stat().st_size, sha256=sha256_file(p)) for p in files]
    processing_code_sha256=sha256_file(__file__)
    libraries={name:importlib.metadata.version(name) for name in ('pandas','pyarrow','duckdb','ijson')}
    out_dir.mkdir(parents=True, exist_ok=True)
    old_path=out_dir/'splits.json'
    old=json.loads(old_path.read_text()) if old_path.exists() else {}
    if as_of is None and old.get('inputs')==inputs and old.get('processing_code_sha256')==processing_code_sha256:
        as_of=old.get('date_upper_bound')
    date_upper_bound=pd.Timestamp(as_of) if as_of else pd.Timestamp.now(tz='UTC')
    if date_upper_bound.tzinfo is None: date_upper_bound=date_upper_bound.tz_localize('UTC')
    date_upper_bound=date_upper_bound.tz_convert('UTC').isoformat()
    identity = dict(protocol=PROTOCOL, inputs=inputs, top_k=top_k, ratios=[train_ratio, val_ratio, test_ratio],
                    processing_code_sha256=processing_code_sha256,libraries=libraries,date_upper_bound=date_upper_bound)
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest_dir = Path(manifests_dir) if manifests_dir else out_dir.parent/'manifests'
    if old.get('signature') == signature and len(old.get('output_hashes', {})) == 2 and all(
          (out_dir/name).exists() and sha256_file(out_dir/name)==digest for name,digest in old['output_hashes'].items()):
        manifest_dir.mkdir(parents=True,exist_ok=True)
        temporary=manifest_dir/'dataset_audit.json.tmp'
        temporary.write_text(json.dumps(old,indent=2));os.replace(temporary,manifest_dir/'dataset_audit.json')
        return old
    staging = Path(tempfile.mkdtemp(prefix='.preprocess-', dir=out_dir.parent))
    reasons, quality = Counter(), Counter()
    raw_labels, raw_projects, field_types = Counter(), Counter(), {}
    schema_fields = ('repo_name','commit_date','is_vul','func_before','func','cve_id','commit_hash','cwe_ids','diff_func')
    parsed = accepted = 0
    con = None
    try:
        batch, parts = [], []
        def flush():
            if batch:
                part = staging / f'part{len(parts)}.parquet'
                pd.DataFrame(batch).to_parquet(part, index=False)
                parts.append(str(part))
                batch.clear()
        for path in files:
            for record in iter_records(path):
                parsed += 1
                if isinstance(record, dict):
                    label = _parse_label(record.get('is_vul', record.get('label')))
                    raw_labels[str(label) if label is not None else 'invalid'] += 1
                    if label is not None and isinstance(record.get('repo_name',record.get('project')),str):
                        raw_projects[(record.get('repo_name',record.get('project')),label)] += 1
                    for field in schema_fields:
                        field_types.setdefault(field,Counter())['missing' if field not in record else type(record[field]).__name__] += 1
                    quality['missing_dates_raw'] += record.get('commit_date') is None
                row, reason = parse_record(record,date_upper_bound)
                if reason:
                    reasons[reason] += 1
                    continue
                accepted += 1
                quality['short_under_30_chars'] += row['code_chars'] < 30
                quality['long_over_50000_chars'] += row['code_chars'] > 50000
                batch.append(row)
                if len(batch) >= batch_size:
                    flush()
                if parsed % 25000 == 0:
                    print(json.dumps(dict(parsed=parsed,accepted=accepted,rejected=sum(reasons.values()))),flush=True)
        flush()
        if not accepted:
            raise ValueError(f'No usable MegaVul rows. Exclusions: {dict(reasons)}')
        con = duckdb.connect(str(staging / 'work.duckdb'), config={'memory_limit': memory_limit, 'threads': 1})
        con.execute("SET preserve_insertion_order=false")
        con.read_parquet(parts).create_view('raw')
        # Quarantine ALL members of conflicting or cross-project exact-code groups.
        con.execute('''CREATE TABLE bad AS SELECT code_hash FROM raw GROUP BY code_hash
                       HAVING count(DISTINCT label)>1 OR count(DISTINCT project)>1
                       UNION SELECT code_hash FROM raw WHERE code_hash IN
                       (SELECT post_hash FROM raw WHERE label=1)''')
        for name, sql in {
            'conflicting_hashes': 'SELECT count(*) FROM (SELECT code_hash FROM raw GROUP BY code_hash HAVING count(DISTINCT label)>1)',
            'cross_project_hashes': 'SELECT count(*) FROM (SELECT code_hash FROM raw GROUP BY code_hash HAVING count(DISTINCT project)>1)',
            'postfix_overlap_rows': 'SELECT count(*) FROM raw WHERE code_hash IN (SELECT post_hash FROM raw WHERE label=1)',
            'quarantined_rows': 'SELECT count(*) FROM raw WHERE code_hash IN (SELECT code_hash FROM bad)',
            'missing_cve': "SELECT count(*) FROM raw WHERE cve_id=''",
            'missing_commit': "SELECT count(*) FROM raw WHERE commit_hash=''",
            'cve_groups_multiple_rows': "SELECT count(*) FROM (SELECT cve_id FROM raw WHERE cve_id!='' GROUP BY cve_id HAVING count(*)>1)",
            'commit_groups_multiple_rows': "SELECT count(*) FROM (SELECT commit_hash FROM raw WHERE commit_hash!='' GROUP BY commit_hash HAVING count(*)>1)",
            'cve_groups_across_projects': "SELECT count(*) FROM (SELECT cve_id FROM raw WHERE cve_id!='' GROUP BY cve_id HAVING count(DISTINCT project)>1)",
            'commit_groups_across_projects': "SELECT count(*) FROM (SELECT commit_hash FROM raw WHERE commit_hash!='' GROUP BY commit_hash HAVING count(DISTINCT project)>1)",
        }.items():
            quality[name] = con.execute(sql).fetchone()[0]
        # Duplicate code carrying different provenance cannot be safely collapsed.
        # Its full provenance is used by later group purging via duplicate_members.
        con.execute('''CREATE TABLE duplicate_members AS SELECT code_hash, cve_id, commit_hash FROM raw''')
        con.execute('''CREATE TABLE clean AS SELECT * FROM raw WHERE code_hash NOT IN (SELECT code_hash FROM bad)
                       QUALIFY row_number() OVER (PARTITION BY code_hash ORDER BY commit_date, project, cve_id, commit_hash)=1''')
        n = con.execute('SELECT count(*) FROM clean').fetchone()[0]
        quality['duplicate_rows_removed'] = accepted - quality['quarantined_rows'] - n
        if n < 3:
            raise ValueError('Insufficient clean data after quarantine')
        # Retain all provenance aliases for overlap checks; no lost CVE/commit edges.
        con.execute('''ALTER TABLE clean ADD COLUMN provenance VARCHAR''')
        con.execute("""CREATE TABLE aliases AS SELECT code_hash,
            to_json(struct_pack(cve_id := coalesce(list(DISTINCT cve_id) FILTER (WHERE cve_id!=''), []::VARCHAR[]),
                                commit_hash := coalesce(list(DISTINCT commit_hash) FILTER (WHERE commit_hash!=''), []::VARCHAR[]))) AS ids
            FROM duplicate_members GROUP BY code_hash""")
        con.execute('UPDATE clean SET provenance=a.ids FROM aliases a WHERE clean.code_hash=a.code_hash')
        top = con.execute('SELECT project, count(*) n FROM clean GROUP BY project ORDER BY n DESC, project LIMIT ?', [top_k]).fetchall()
        names = [p for p,_ in top]
        con.execute('CREATE TABLE top_projects(project VARCHAR)')
        con.executemany('INSERT INTO top_projects VALUES (?)', [(p,) for p in names])
        columns = 'func,label,project,commit_date,cve_id,commit_hash,code_hash,provenance'
        con.execute(f'COPY (SELECT {columns} FROM clean WHERE project IN (SELECT project FROM top_projects) ORDER BY commit_date,project,code_hash) TO ? (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 2048)', [str(staging/'top10.parquet')])
        con.execute(f'COPY (SELECT {columns} FROM clean WHERE project NOT IN (SELECT project FROM top_projects) ORDER BY commit_date,project,code_hash) TO ? (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 2048)', [str(staging/'tail.parquet')])
        distribution = con.execute('SELECT project, label, count(*) FROM clean GROUP BY project,label ORDER BY project,label').fetchall()
        # Bounded approximate near-duplicate audit: candidate prefix of token shingles.
        # This is diagnostic only; it is NOT a proof that near-duplicates are absent.
        quality['near_duplicate_audit'] = 'not_exhaustive; exact normalized code enforced; run audit_data.py for bounded lexical candidates'
        result = dict(**identity, signature=signature, raw_records_parsed=parsed, usable_records=accepted,
                      rejected_records=sum(reasons.values()), exclusions=dict(reasons), quality=dict(quality),
                      raw_label_counts=dict(raw_labels), raw_schema_types={key:dict(value) for key,value in field_types.items()},
                      raw_project_class_counts=[dict(project=p,label=label,n=count) for (p,label),count in sorted(raw_projects.items())],
                      rows_after_validation_and_dedup=n, top_project_counts=dict(top),
                      class_distribution=[dict(project=p,label=y,n=c) for p,y,c in distribution],
                      tail_rows=n-sum(c for _,c in top),
                      train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
                      split_policy='LOPO source temporal ratios rescaled to train:val; equal timestamps indivisible; all CVE/commit groups crossing partitions purged; held-out early DA/later test shared by methods',
                      function_column_policy='positive=func_before only; negative=func only; no post-fix augmentation',
                      output_hashes={name:sha256_file(staging/name) for name in ('top10.parquet','tail.parquet')},
                      source_url=SOURCE, specification_url=SPEC,
                      generated_at=datetime.now(timezone.utc).isoformat(),
                      release='user supplied; verify acquisition.json before claiming authentic release',
                      license='Upstream repository GPL-3.0; function code retains original project licensing; verify release terms')
        # splits.json is the commit marker; consumers validate its output digests.
        (staging/'splits.json').write_text(json.dumps(result, indent=2))
        for name in ('top10.parquet','tail.parquet','splits.json'):
            os.replace(staging/name, out_dir/name)
        manifest_dir = Path(manifests_dir) if manifests_dir else out_dir.parent/'manifests'
        manifest_dir.mkdir(parents=True, exist_ok=True)
        temp_manifest = manifest_dir/'dataset_audit.json.tmp'
        temp_manifest.write_text(json.dumps(result, indent=2))
        os.replace(temp_manifest, manifest_dir/'dataset_audit.json')
        return result
    finally:
        if con:
            con.close()
        shutil.rmtree(staging)
