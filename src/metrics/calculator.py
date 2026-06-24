"""
Implementation of all quantitative metrics defined in the manuscript (Tables 1-9).

Author: Amit Pimpalkar
Organization: RBU, Nagpur
Year: 2026
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def compute_accuracy(predictions: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> float:
    """Overall learning accuracy (Table 1)."""
    pred_binary = (predictions > threshold).astype(int)
    return np.mean(pred_binary == targets)


def compute_demographic_disparity(predictions: np.ndarray, targets: np.ndarray,
                                  demographic_groups: np.ndarray) -> Dict[str, float]:
    """
    Compute demographic learning gain disparity (Table 2).
    Returns the disparity as max-min difference across groups.
    """
    group_accuracies = {}
    for group_id in np.unique(demographic_groups):
        mask = demographic_groups == group_id
        if np.sum(mask) > 0:
            group_accuracies[group_id] = compute_accuracy(predictions[mask], targets[mask])
    
    if not group_accuracies:
        return {'disparity': 0.0, 'max': 0.0, 'min': 0.0}
    
    acc_values = list(group_accuracies.values())
    return {
        'disparity': max(acc_values) - min(acc_values),
        'max': max(acc_values),
        'min': min(acc_values),
        'per_group': group_accuracies
    }


def compute_equity_stability_index(equity_loss_trajectory: np.ndarray) -> float:
    """
    Temporal volatility of equity measurements (Table 4). Lower is better.
    Computed as the standard deviation of the first-order differences.
    """
    if len(equity_loss_trajectory) < 2:
        return 0.0
    diffs = np.diff(equity_loss_trajectory)
    return np.std(diffs)


def compute_counterfactual_divergence(factual_outcomes: np.ndarray,
                                      counterfactual_outcomes: np.ndarray) -> float:
    """
    Mean Absolute Error between factual and counterfactual trajectories (Table 5).
    """
    return np.mean(np.abs(factual_outcomes - counterfactual_outcomes))


def compute_attention_alignment(attention_weights: np.ndarray,
                                equity_residuals: np.ndarray,
                                threshold: float = 0.01) -> float:
    """
    Correlation between attention focus and equity risk regions (Table 6).
    Higher is better.
    """
    # Flatten and normalize
    att_flat = attention_weights.flatten()
    eq_flat = (equity_residuals.flatten() > threshold).astype(float)
    
    if np.std(att_flat) == 0 or np.std(eq_flat) == 0:
        return 0.5  # Neutral score
    
    # Compute Pearson correlation
    correlation = np.corrcoef(att_flat, eq_flat)[0, 1]
    # Map from [-1, 1] to [0, 1] for scoring
    return (correlation + 1) / 2


def compute_pareto_balance_score(loss_history: Dict[str, List[float]]) -> float:
    """
    Stability of multi-objective trade-offs (Table 7).
    Lower variance across objectives indicates better balance.
    """
    if not loss_history:
        return 0.0
    
    # Compute coefficient of variation (CV) for each objective
    cvs = []
    for objective, values in loss_history.items():
        values = np.array(values)
        if np.mean(values) > 0:
            cv = np.std(values) / np.mean(values)
            cvs.append(cv)
    
    if not cvs:
        return 0.0
    
    # Balance score = 1 - average CV (capped at 0)
    avg_cv = np.mean(cvs)
    return max(0.0, 1.0 - avg_cv)


def compute_equity_retention_ratio(student_equity_loss: float,
                                   teacher_equity_loss: float) -> float:
    """
    Equity preservation after distillation (Table 8).
    """
    if teacher_equity_loss == 0:
        return 1.0
    return max(0.0, 1.0 - abs(student_equity_loss - teacher_equity_loss) / teacher_equity_loss)


def compute_composite_score(accuracy: float, equity_metric: float,
                            engagement: float, weights: Tuple[float, float, float] = (0.4, 0.35, 0.25)) -> float:
    """
    Weighted composite score (Table 9).
    equity_metric here should be normalized (e.g., 1 - disparity or 1 - stability_index).
    """
    return weights[0] * accuracy + weights[1] * equity_metric + weights[2] * engagement


def compute_all_metrics(predictions: np.ndarray,
                        targets: np.ndarray,
                        demographic_groups: np.ndarray,
                        factual_outcomes: np.ndarray,
                        counterfactual_outcomes: np.ndarray,
                        attention_weights: np.ndarray,
                        equity_residuals: np.ndarray,
                        engagement_scores: np.ndarray,
                        loss_trajectory: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Compute all metrics in a single call for convenience.
    """
    acc = compute_accuracy(predictions, targets)
    disparity_result = compute_demographic_disparity(predictions, targets, demographic_groups)
    disparity = disparity_result['disparity']
    
    # For composite, we use (1 - normalized_disparity) as the equity score
    max_possible_disparity = 1.0
    equity_score = max(0.0, 1.0 - (disparity / max_possible_disparity))
    
    stability = compute_equity_stability_index(np.array(loss_trajectory.get('equity', [0])))
    # Stability score: lower is better, so invert for composite (max 1)
    stability_score = max(0.0, 1.0 - stability)
    
    divergence = compute_counterfactual_divergence(factual_outcomes, counterfactual_outcomes)
    # Divergence score: lower is better, invert
    divergence_score = max(0.0, 1.0 - divergence)
    
    alignment = compute_attention_alignment(attention_weights, equity_residuals)
    balance = compute_pareto_balance_score(loss_trajectory)
    
    # Engagement retention proxy
    engagement_retention = np.mean(engagement_scores)
    
    # Composite (using accuracy, equity_score, engagement)
    composite = compute_composite_score(acc, equity_score, engagement_retention)
    
    return {
        'accuracy': acc,
        'demographic_disparity': disparity,
        'equity_stability_index': stability,
        'counterfactual_divergence': divergence,
        'attention_alignment': alignment,
        'pareto_balance_score': balance,
        'engagement_retention': engagement_retention,
        'composite_score': composite,
        'equity_score': equity_score,
        'stability_score': stability_score,
        'divergence_score': divergence_score
    }