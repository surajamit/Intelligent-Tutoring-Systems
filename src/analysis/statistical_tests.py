"""
Statistical significance testing for model comparisons.

Author: Amit Pimpalkar
Organization: RBU, Nagpur
Year: 2026

Implements paired t-tests and Wilcoxon signed-rank tests, Validated Hyperparameter & Baseline Detailed Analysis.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional


def paired_t_test(scores_model: np.ndarray, scores_baseline: np.ndarray,
                  alpha: float = 0.05) -> Dict[str, float]:
    """
    Perform paired t-test between two sets of scores.

    Args:
        scores_model: Array of scores from the proposed model
        scores_baseline: Array of scores from the baseline model
        alpha: Significance level

    Returns:
        Dictionary with t-statistic, p-value, and significance flag
    """
    t_stat, p_value = stats.ttest_rel(scores_model, scores_baseline)
    significant = p_value < alpha
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': significant,
        'alpha': alpha
    }


def wilcoxon_signed_rank_test(scores_model: np.ndarray, scores_baseline: np.ndarray,
                              alpha: float = 0.05) -> Dict[str, float]:
    """
    Perform Wilcoxon signed-rank test for non-normal distributions.

    Args:
        scores_model: Array of scores from the proposed model
        scores_baseline: Array of scores from the baseline model
        alpha: Significance level

    Returns:
        Dictionary with statistic, p-value, and significance flag
    """
    w_stat, p_value = stats.wilcoxon(scores_model, scores_baseline)
    significant = p_value < alpha
    return {
        'statistic': w_stat,
        'p_value': p_value,
        'significant': significant,
        'alpha': alpha
    }


def compute_effect_size(scores_model: np.ndarray, scores_baseline: np.ndarray,
                        method: str = 'cohen_d') -> float:
    """
    Compute effect size (Cohen's d or Hedges' g).

    Args:
        scores_model: Array of scores from the proposed model
        scores_baseline: Array of scores from the baseline model
        method: 'cohen_d' or 'hedges_g'

    Returns:
        Effect size value
    """
    n1 = len(scores_model)
    n2 = len(scores_baseline)
    mean1 = np.mean(scores_model)
    mean2 = np.mean(scores_baseline)
    var1 = np.var(scores_model, ddof=1)
    var2 = np.var(scores_baseline, ddof=1)

    if method == 'cohen_d':
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        d = (mean1 - mean2) / np.sqrt(pooled_var)
        return d
    elif method == 'hedges_g':
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        g = (mean1 - mean2) / np.sqrt(pooled_var)
        # Correct for small sample bias
        correction = 1 - 3 / (4 * (n1 + n2) - 9)
        return g * correction
    else:
        raise ValueError("method must be 'cohen_d' or 'hedges_g'")


def run_all_significance_tests(model_scores: Dict[str, np.ndarray],
                               baseline_scores: Dict[str, np.ndarray],
                               metric_name: str) -> Dict[str, Dict]:
    """
    Run both t-test and Wilcoxon test for each baseline comparison.

    Args:
        model_scores: Dictionary mapping dataset to model score array
        baseline_scores: Dictionary mapping baseline name to dict of dataset scores
        metric_name: Name of the metric (for logging)

    Returns:
        Nested dictionary with test results for each baseline and dataset
    """
    results = {}
    for baseline_name, baseline_data in baseline_scores.items():
        results[baseline_name] = {}
        for dataset, scores_model in model_scores.items():
            if dataset not in baseline_data:
                continue
            scores_baseline = baseline_data[dataset]
            t_test = paired_t_test(scores_model, scores_baseline)
            wilcoxon = wilcoxon_signed_rank_test(scores_model, scores_baseline)
            effect = compute_effect_size(scores_model, scores_baseline)
            results[baseline_name][dataset] = {
                't_test': t_test,
                'wilcoxon': wilcoxon,
                'effect_size': effect
            }
    return results
