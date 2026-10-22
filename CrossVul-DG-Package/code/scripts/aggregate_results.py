"""Compatibility launcher; canonical entry point is scripts/aggregate_results.py."""
from pathlib import Path
import os
import runpy
ROOT=Path(__file__).resolve().parents[3]
os.chdir(ROOT)
runpy.run_path(str(ROOT/'scripts'/'aggregate_results.py'),run_name='__main__')
