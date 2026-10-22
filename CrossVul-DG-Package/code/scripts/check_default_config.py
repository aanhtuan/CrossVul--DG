"""Compatibility launcher; canonical entry point is scripts/check_default_config.py."""
from pathlib import Path
import os
import runpy
ROOT=Path(__file__).resolve().parents[3]
os.chdir(ROOT)
runpy.run_path(str(ROOT/'scripts'/'check_default_config.py'),run_name='__main__')
