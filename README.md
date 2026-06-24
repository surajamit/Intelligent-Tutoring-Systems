# Multi-Objective Deep Optimization for Equity-Aware Intelligent Tutoring Systems

This repository implements an integrated framework for simultaneously optimizing learning accuracy, demographic equity, and learner engagement in AI-driven educational platforms. The architecture treats fairness as a first-class optimization signal rather than a post-hoc correction mechanism.

## Project Overview

Educational AI systems increasingly influence curriculum pacing, learner confidence, and academic trajectories. Traditional optimization approaches prioritize prediction accuracy while inadvertently encoding demographic biases present in training data. This work addresses the challenge through:

- **Demographic Sensitivity Gradient Encoding (DSGE)**: Quantifies how demographic attributes influence learning predictions at the gradient level
- **Counterfactual Equity Replay Networks (CERN)**: Models alternative learning trajectories under demographic interventions
- **Engagement-Weighted Fairness Attention Fusion (EWFAF)**: Balances engagement optimization with equity preservation
- **Pareto-Adaptive Multi-Objective Controller**: Dynamically adjusts objective weights to maintain systemic balance
- **Equity-Preserving Policy Distillation**: Compresses models while retaining fairness guarantees

## Installation

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd equity_aware_tutoring_system
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt