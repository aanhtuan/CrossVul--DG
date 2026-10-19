from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


PATH = Path(r"D:\Datawarehouse\megavul.json")
CHUNK_SIZE = 1024 * 1024


def normalize_project(value) -> str | None:
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


def iter_json_array(path: Path):
    decoder = json.JSONDecoder()
    buffer = ""
    started = False

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk and not buffer:
                break
            buffer += chunk

            while True:
                buffer = buffer.lstrip()

                if not started:
                    if not buffer:
                        break
                    if buffer[0] != "[":
                        raise ValueError(f"Expected JSON array in {path}")
                    buffer = buffer[1:]
                    started = True
                    continue

                buffer = buffer.lstrip()
                if buffer.startswith("]"):
                    return
                if buffer.startswith(","):
                    buffer = buffer[1:]
                    continue
                if not buffer:
                    break

                try:
                    item, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if not chunk:
                        raise
                    break

                yield item
                buffer = buffer[end:]

            if not chunk:
                break


def main() -> None:
    counts = Counter()
    total_rows = 0

    for row in iter_json_array(PATH):
        total_rows += 1
        project = normalize_project(
            row.get("repo_name") or row.get("project") or row.get("repo")
        )
        if project:
            counts[project] += 1

        if total_rows % 100000 == 0:
            print(f"Read {total_rows:,} rows...", flush=True)

    print(f"Total rows: {total_rows:,}")
    print(f"Total projects: {len(counts):,}")
    print()

    for project, n in counts.most_common(100):
        print(f"{project}\t{n}")


if __name__ == "__main__":
    main()
