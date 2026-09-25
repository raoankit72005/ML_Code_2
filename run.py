#!/usr/bin/env python3
"""Run this file from a terminal or use !python run.py ... in a notebook."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from er_pipeline.cli import main

if __name__ == '__main__':
    main()
