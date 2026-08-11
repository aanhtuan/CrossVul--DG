"""Preprocess MegaVul into top-K and tail parquet files without loading
all raw JSON into RAM.

Pipeline:
1. Stream JSON/JSONL records.
2. Write validated rows to small parquet parts.
3. Optionally deduplicate normalized code, keeping the earliest occurrence.
4. Select top-K projects by post-deduplication row count.
5. Compute chronological 80:10:10 cut-offs.
6. Write top10.parquet, tail.parquet, and splits.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterator

import ijson
import pandas as pd

try:
    import duckdb
except ImportError as exc:
    raise ImportError(
        "Missing duckdb. Install with: pip install duckdb ijson pyarrow"
    ) from exc


def _first_present(record: dict[str, Any], keys: list[str]) -> Any | None:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _parse_label(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value != 0)

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "vulnerable", "vul", "positive", "unsafe"}:
        return 1
    if text in {"0", "false", "no", "safe", "non-vulnerable", "non_vulnerable", "negative", "benign"}:
        return 0
    return None


def _project_name(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("name") or value.get("full_name") or value.get("path") or value.get("url")
    if value is None:
        return None

    project = str(value).strip().rstrip("/")
    if not project:
        return None
    if "/" in project:
        project = project.split("/")[-1]
    if project.endswith(".git"):
        project = project[:-4]
    return project or None

def _parse_commit_date(value):
    """Parse string dates and Unix timestamps with automatic unit detection."""
    if value is None:
        return None

    # Numeric or numeric string
    try:
        number = float(value)

        if number > 1e17:
            unit = "ns"
        elif number > 1e14:
            unit = "us"
        elif number > 1e11:
            unit = "ms"
        else:
            unit = "s"

        parsed = pd.to_datetime(
            number,
            unit=unit,
            errors="coerce",
            utc=True
        )

        if not pd.isna(parsed):
            return parsed

    except (TypeError, ValueError, OverflowError):
        pass

    # Normal ISO/date string
    parsed = pd.to_datetime(
        value,
        errors="coerce",
        utc=True
    )

    if pd.isna(parsed):
        return None

    return parsed
def _row_from_dict(record: dict[str, Any]) -> dict[str, Any] | None:
    func = _first_present(record, [
        "func", "func_before", "function", "code", "function_before",
        "before_func", "before_code",
    ])
    label = _parse_label(_first_present(record, [
        "label", "is_vul", "vul", "vulnerable", "target", "is_vulnerable",
    ]))
    project = _project_name(_first_present(record, [
        "project", "repo", "repository", "project_name",
        "repo_name", "repository_name",
    ]))
    commit_date = _first_present(record, [
        "commit_date", "date", "commit_time", "authored_date",
        "commit_datetime", "commit_timestamp", "author_date", "committer_date",
    ])

    if func is None or label is None or project is None or commit_date is None:
        return None

    func = str(func).strip()
    if not func:
        return None

    parsed_date = _parse_commit_date(commit_date)
    if parsed_date is None or pd.isna(parsed_date):
        return None

    return {
        "func": func,
        "label": int(label),
        "project": project,
        "commit_date": parsed_date,
    }


_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", flags=re.DOTALL)
_LINE_COMMENT = re.compile(r"//.*?$", flags=re.MULTILINE)
_WHITESPACE = re.compile(r"\s+")


def _normalize_code(code: str) -> str:
    code = _BLOCK_COMMENT.sub(" ", str(code))
    code = _LINE_COMMENT.sub(" ", code)
    code = _WHITESPACE.sub(" ", code)
    return code.strip()


def _code_hash(code: str) -> str:
    normalized = _normalize_code(code)
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()


def _detect_json_array(file_path: Path) -> bool:
    with file_path.open("rb") as handle:
        while True:
            char = handle.read(1)
            if not char:
                return False
            if not char.isspace():
                return char == b"["


def _iter_records(file_path: Path) -> Iterator[dict[str, Any]]:
    if _detect_json_array(file_path):
        with file_path.open("rb") as handle:
            for record in ijson.items(handle, "item"):
                if isinstance(record, dict):
                    yield record
    else:
        with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    yield record


def _find_raw_files(raw_dir: Path) -> list[Path]:
    files = sorted(set(raw_dir.rglob("*.json")) | set(raw_dir.rglob("*.jsonl")))
    return [path for path in files if path.is_file()]


def _write_batch(
    batch: list[dict[str, Any]],
    parts_dir: Path,
    part_index: int,
    add_hash: bool,
) -> Path:
    frame = pd.DataFrame(batch)
    if add_hash:
        frame["code_hash"] = frame["func"].map(_code_hash)
    part_path = parts_dir / f"part_{part_index:06d}.parquet"
    frame.to_parquet(part_path, index=False)
    return part_path


def _stream_to_parts(
    raw_dir: Path,
    parts_dir: Path,
    batch_size: int,
    add_hash: bool,
) -> dict[str, int]:
    raw_files = _find_raw_files(raw_dir)
    if not raw_files:
        raise FileNotFoundError(f"No JSON/JSONL files found under: {raw_dir}")

    parts_dir.mkdir(parents=True, exist_ok=True)
    batch: list[dict[str, Any]] = []
    parsed = accepted = rejected = part_index = 0

    print(f"Found {len(raw_files)} raw JSON file(s).", flush=True)

    for file_path in raw_files:
        print(f"Reading: {file_path}", flush=True)
        for record in _iter_records(file_path):
            parsed += 1
            row = _row_from_dict(record)
            if row is None:
                rejected += 1
            else:
                batch.append(row)
                accepted += 1

            if len(batch) >= batch_size:
                path = _write_batch(batch, parts_dir, part_index, add_hash)
                print(
                    f"  Wrote {path.name}: {len(batch):,} rows; "
                    f"parsed={parsed:,}, accepted={accepted:,}.",
                    flush=True,
                )
                batch.clear()
                part_index += 1

            if parsed % 10_000 == 0:
                print(
                    f"  Progress: parsed={parsed:,}, accepted={accepted:,}, rejected={rejected:,}.",
                    flush=True,
                )

    if batch:
        path = _write_batch(batch, parts_dir, part_index, add_hash)
        print(f"  Wrote {path.name}: {len(batch):,} rows.", flush=True)
        batch.clear()
        part_index += 1

    if accepted == 0:
        raise RuntimeError(
            "Parsed zero usable rows. Inspect one raw record and update _row_from_dict()."
        )

    print(
        f"Finished parsing: {parsed:,} parsed, {accepted:,} accepted, "
        f"{rejected:,} rejected, {part_index} parts.",
        flush=True,
    )
    return {
        "raw_records_parsed": parsed,
        "usable_records": accepted,
        "rejected_records": rejected,
        "temporary_parts": part_index,
    }


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _process_parts(
    parts_dir: Path,
    out_dir: Path,
    top_k: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    dedup: bool,
    stats: dict[str, int],
) -> None:
    parquet_glob = _sql_path(parts_dir / "part_*.parquet")
    top_path = out_dir / "top10.parquet"
    tail_path = out_dir / "tail.parquet"
    splits_path = out_dir / "splits.json"

    con = duckdb.connect(str(out_dir / "_preprocess.duckdb"))
    try:
        con.execute("SET memory_limit='6GB'")
        con.execute(f"SET temp_directory='{_sql_path(out_dir / '_duckdb_tmp')}'")
        con.execute("SET preserve_insertion_order=false")

        hash_select = ", CAST(code_hash AS VARCHAR) AS code_hash" if dedup else ""
        con.execute(f"""
            CREATE OR REPLACE VIEW raw_parts AS
            SELECT
                CAST(func AS VARCHAR) AS func,
                CAST(label AS INTEGER) AS label,
                CAST(project AS VARCHAR) AS project,
                CAST(commit_date AS TIMESTAMPTZ) AS commit_date
                {hash_select}
            FROM read_parquet('{parquet_glob}', union_by_name=true)
            WHERE func IS NOT NULL
              AND trim(func) <> ''
              AND project IS NOT NULL
              AND trim(project) <> ''
              AND commit_date IS NOT NULL
              AND label IN (0, 1)
        """)

        if dedup:
            con.execute("""
                CREATE OR REPLACE TABLE clean_data AS
                SELECT func, label, project, commit_date
                FROM raw_parts
                QUALIFY row_number() OVER (
                    PARTITION BY code_hash
                    ORDER BY commit_date ASC, project ASC
                ) = 1
            """)
        else:
            con.execute("""
                CREATE OR REPLACE TABLE clean_data AS
                SELECT func, label, project, commit_date
                FROM raw_parts
            """)

        clean_count = con.execute("SELECT count(*) FROM clean_data").fetchone()[0]
        if clean_count == 0:
            raise RuntimeError("No usable records remain after validation/deduplication.")

        con.execute(f"""
            CREATE OR REPLACE TABLE top_projects AS
            SELECT project
            FROM clean_data
            GROUP BY project
            ORDER BY count(*) DESC, project ASC
            LIMIT {int(top_k)}
        """)

        con.execute("""
            CREATE OR REPLACE VIEW top_data AS
            SELECT c.*
            FROM clean_data c
            INNER JOIN top_projects p USING (project)
        """)
        con.execute("""
            CREATE OR REPLACE VIEW tail_data AS
            SELECT c.*
            FROM clean_data c
            LEFT JOIN top_projects p USING (project)
            WHERE p.project IS NULL
        """)

        top_count = con.execute("SELECT count(*) FROM top_data").fetchone()[0]
        tail_count = con.execute("SELECT count(*) FROM tail_data").fetchone()[0]
        if top_count < 3:
            raise RuntimeError("Not enough top-K rows for chronological splitting.")

        n_train = int(top_count * train_ratio)
        n_val = int(top_count * val_ratio)
        n_test = top_count - n_train - n_val
        if min(n_train, n_val, n_test) <= 0:
            raise RuntimeError("One split is empty. Adjust ratios or use more data.")

        train_cutoff = con.execute(f"""
            SELECT commit_date FROM top_data
            ORDER BY commit_date ASC, project ASC
            LIMIT 1 OFFSET {n_train - 1}
        """).fetchone()[0]
        val_cutoff = con.execute(f"""
            SELECT commit_date FROM top_data
            ORDER BY commit_date ASC, project ASC
            LIMIT 1 OFFSET {n_train + n_val - 1}
        """).fetchone()[0]

        top_counts = con.execute("""
            SELECT project, count(*) AS n
            FROM top_data
            GROUP BY project
            ORDER BY n DESC, project ASC
        """).fetchall()
        tail_projects = con.execute(
            "SELECT count(DISTINCT project) FROM tail_data"
        ).fetchone()[0]

        con.execute(f"""
            COPY (
                SELECT func, label, project, commit_date
                FROM top_data
                ORDER BY commit_date ASC, project ASC
            ) TO '{_sql_path(top_path)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        con.execute(f"""
            COPY (
                SELECT func, label, project, commit_date
                FROM tail_data
                ORDER BY commit_date ASC, project ASC
            ) TO '{_sql_path(tail_path)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        splits = {
            **stats,
            "dedup_enabled": dedup,
            "rows_after_validation_and_dedup": int(clean_count),
            "top_k": int(top_k),
            "top_project_counts": {str(p): int(n) for p, n in top_counts},
            "tail_rows": int(tail_count),
            "tail_projects": int(tail_projects),
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "test_ratio": float(test_ratio),
            "n_train": int(n_train),
            "n_val": int(n_val),
            "n_test": int(n_test),
            "train_cutoff": str(train_cutoff),
            "val_cutoff": str(val_cutoff),
            "split_policy": (
                "Pooled top-K rows sorted by commit_date ascending; "
                "oldest train, middle validation, newest test."
            ),
            "dedup_policy": (
                "SHA-256 of comment-stripped and whitespace-normalized code; "
                "earliest chronological occurrence retained."
                if dedup else "Disabled"
            ),
        }
        with splits_path.open("w", encoding="utf-8") as handle:
            json.dump(splits, handle, indent=2, ensure_ascii=False, default=str)

        print(f"Rows after validation/dedup: {clean_count:,}", flush=True)
        print(f"Top-{top_k} projects: {splits['top_project_counts']}", flush=True)
        print(
            f"Top rows: {top_count:,}; tail rows: {tail_count:,} "
            f"across {tail_projects:,} projects.",
            flush=True,
        )
        print(
            f"Chronological split: train={n_train:,}, val={n_val:,}, test={n_test:,}.",
            flush=True,
        )
        print(f"Wrote {top_path}", flush=True)
        print(f"Wrote {tail_path}", flush=True)
        print(f"Wrote {splits_path}", flush=True)
    finally:
        con.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream and preprocess a large MegaVul JSON snapshot."
    )
    parser.add_argument("--raw_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--dedup", action="store_true")
    parser.add_argument("--keep_temp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(
            "train_ratio + val_ratio + test_ratio must equal 1.0; "
            f"received {ratio_sum:.8f}."
        )
    if args.top_k <= 0:
        raise ValueError("--top_k must be greater than zero.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be greater than zero.")
    if not args.raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {args.raw_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = args.out_dir / "_parts"

    if parts_dir.exists():
        shutil.rmtree(parts_dir)

    for stale_path in [
        args.out_dir / "top10.parquet",
        args.out_dir / "tail.parquet",
        args.out_dir / "splits.json",
        args.out_dir / "_preprocess.duckdb",
        args.out_dir / "_duckdb_tmp",
    ]:
        if stale_path.is_dir():
            shutil.rmtree(stale_path)
        elif stale_path.exists():
            stale_path.unlink()

    try:
        stats = _stream_to_parts(
            raw_dir=args.raw_dir,
            parts_dir=parts_dir,
            batch_size=args.batch_size,
            add_hash=args.dedup,
        )
        _process_parts(
            parts_dir=parts_dir,
            out_dir=args.out_dir,
            top_k=args.top_k,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            dedup=args.dedup,
            stats=stats,
        )
    finally:
        if not args.keep_temp:
            if parts_dir.exists():
                shutil.rmtree(parts_dir)
            duckdb_file = args.out_dir / "_preprocess.duckdb"
            if duckdb_file.exists():
                duckdb_file.unlink()
            duckdb_tmp = args.out_dir / "_duckdb_tmp"
            if duckdb_tmp.exists():
                shutil.rmtree(duckdb_tmp)
            print("Removed temporary preprocessing files.", flush=True)


if __name__ == "__main__":
    main()