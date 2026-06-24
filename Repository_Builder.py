#!/usr/bin/env python
"""
Complete Repository Builder – Final Version
Author: Amit Pimpalkar, RBU Nagpur
Year: 2026

Generates every file of the equity‑aware tutoring system.
Run this script to produce the full repository ZIP.
"""

import os
import shutil
import zipfile
from pathlib import Path

def write_file(filepath: str, content: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content.lstrip())

def build():
    base = "equity_aware_tutoring_system"
    if os.path.exists(base): shutil.rmtree(base)
    print(f"Building repository: {base}")

    # ------------------------------------------------------------
    # 1. ROOT FILES
    # ------------------------------------------------------------
    write_file(f"{base}/README.md", """# Multi-Objective Deep Optimization for Equity-Aware Intelligent Tutoring Systems

This repository implements an integrated framework for simultaneously optimizing learning accuracy, demographic equity, and learner engagement in AI-driven educational platforms. The architecture treats fairness as a first‑class optimization signal rather than a post‑hoc correction.

## Key Innovations
- **Demographic Sensitivity Gradient Encoding (DSGE)**: Quantifies demographic influence at the gradient level.
- **Counterfactual Equity Replay Networks (CERN)**: Models alternative learning trajectories under demographic interventions.
- **Engagement‑Weighted Fairness Attention Fusion (EWFAF)**: Balances engagement optimisation with equity preservation.
- **Pareto‑Adaptive Multi‑Objective Controller**: Dynamically adjusts objective weights.
- **Equity‑Preserving Policy Distillation (EPPDV)**: Compresses the model while retaining fairness guarantees.

## Quick Start
```bash
conda env create -f environment.yml
conda activate equity_tutor
make data-prep
make train
make evaluate
make distill
make serve