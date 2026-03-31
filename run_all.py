#!/usr/bin/env python3
"""
run_all.py
----------
One-click setup:  generate data → train → generate PDF report → launch Streamlit
Run with:   python run_all.py
"""

import os, sys, subprocess

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)

steps = [
    ("Generating datasets",       [sys.executable, "generate_datasets.py"]),
    ("Training models",           [sys.executable, "train_models.py"]),
    ("Generating PDF report",     [sys.executable, "generate_report.py"]),
]

for label, cmd in steps:
    print(f"\n{'='*60}")
    print(f"  {label}…")
    print('='*60)
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"ERROR: {label} failed. Aborting.")
        sys.exit(1)

print("\n" + "="*60)
print("  Launching Streamlit app…  (Ctrl+C to stop)")
print("="*60)
subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
                "--server.port", "8501"])
