"""Compatibility launcher; canonical entry point is scripts/prepare_data.py."""
from pathlib import Path
import os
import runpy
ROOT=Path(__file__).resolve().parents[3]
os.chdir(ROOT)
runpy.run_path(str(ROOT/'scripts'/'prepare_data.py'),run_name='__main__')
