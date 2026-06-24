"""
Analysis and visualization modules for the equity-aware tutoring system.

Author: Amit Pimpalkar
Organization: RBU, Nagpur
Year: 2026
"""

from .statistical_tests import (
    paired_t_test,
    wilcoxon_signed_rank_test,
    compute_effect_size,
    run_all_significance_tests
)
from .visualization import (
    plot_accuracy_comparison,
    plot_disparity_reduction,
    plot_engagement_retention,
    plot_equity_stability,
    plot_counterfactual_divergence,
    plot_attention_alignment,
    plot_pareto_balance,
    plot_policy_retention,
    plot_composite_scores,
    generate_all_figures
)

__all__ = [
    'paired_t_test',
    'wilcoxon_signed_rank_test',
    'compute_effect_size',
    'run_all_significance_tests',
    'plot_accuracy_comparison',
    'plot_disparity_reduction',
    'plot_engagement_retention',
    'plot_equity_stability',
    'plot_counterfactual_divergence',
    'plot_attention_alignment',
    'plot_pareto_balance',
    'plot_policy_retention',
    'plot_composite_scores',
    'generate_all_figures'
]