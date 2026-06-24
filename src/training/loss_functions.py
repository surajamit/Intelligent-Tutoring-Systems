"""
Loss function definitions.

Author: Amit Pimpalkar
Organization: RBU, Nagpur
Year: 2026
"""

import torch
import torch.nn as nn


def compute_accuracy_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy for mastery prediction."""
    return nn.BCELoss()(predictions, targets)


def compute_engagement_loss(engagement_scores: torch.Tensor) -> torch.Tensor:
    """Engagement retention loss (minimize disengagement)."""
    return (1.0 - engagement_scores).mean()