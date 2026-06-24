"""
Centralized metrics computation module for the equity-aware tutoring system.

Author: Amit Pimpalkar
Organization: RBU, Nagpur
Year: 2026
"""

from .calculator import (
    compute_accuracy,
    compute_demographic_disparity,
    compute_equity_stability_index,
    compute_counterfactual_divergence,
    compute_attention_alignment,
    compute_pareto_balance_score,
    compute_equity_retention_ratio,
    compute_composite_score,
    compute_all_metrics
)

__all__ = [
    'compute_accuracy',
    'compute_demographic_disparity',
    'compute_equity_stability_index',
    'compute_counterfactual_divergence',
    'compute_attention_alignment',
    'compute_pareto_balance_score',
    'compute_equity_retention_ratio',
    'compute_composite_score',
    'compute_all_metrics'
]