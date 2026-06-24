"""
Pareto-Adaptive Multi-Objective Controller (PA3EO).

Author: Amit Pimpalkar
Organization: RBU, Nagpur
Year: 2026

Dynamically balances accuracy, equity, and engagement objectives by adapting
loss weights based on variance across objectives.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ParetoAdaptiveObjectiveController:
    """
    Manages dynamic weight adjustment for multi-objective optimization.
    
    Rather than fixed weights, this controller:
    1. Monitors variance across accuracy, equity, and engagement losses
    2. Increases weights for objectives showing high variance (instability)
    3. Maintains weights on probability simplex via projection
    4. Enforces minimum threshold to prevent objective collapse
    """
    
    def __init__(
        self,
        initial_weights: torch.Tensor = None,
        meta_learning_rate: float = 0.001,
        weight_lower_bound: float = 0.05,
        update_frequency: int = 5
    ):
        """
        Initialize Pareto controller.
        
        Args:
            initial_weights: Starting weights [accuracy, equity, engagement]
            meta_learning_rate: Step size for weight updates
            weight_lower_bound: Minimum allowable weight per objective
            update_frequency: Update weights every N epochs
        """
        if initial_weights is None:
            initial_weights = torch.tensor([0.40, 0.35, 0.25])
        
        self.weights = initial_weights.clone().detach()
        self.meta_lr = meta_learning_rate
        self.lower_bound = weight_lower_bound
        self.update_freq = update_frequency
        
        self.loss_history = {
            'accuracy': [],
            'equity': [],
            'engagement': []
        }
        
        logger.info(
            f"Initialized PA3EO with weights={self.weights.tolist()}, "
            f"meta_lr={meta_learning_rate}"
        )
    
    def update_weights(
        self,
        loss_accuracy: float,
        loss_equity: float,
        loss_engagement: float,
        epoch: int
    ):
        """
        Update objective weights based on loss variance.
        
        Args:
            loss_accuracy: Current accuracy loss value
            loss_equity: Current equity loss value
            loss_engagement: Current engagement loss value
            epoch: Current training epoch
        """
        self.loss_history['accuracy'].append(loss_accuracy)
        self.loss_history['equity'].append(loss_equity)
        self.loss_history['engagement'].append(loss_engagement)
        
        if epoch % self.update_freq != 0:
            return
        
        window_size = min(10, len(self.loss_history['accuracy']))
        
        var_acc = torch.var(torch.tensor(self.loss_history['accuracy'][-window_size:]))
        var_eq = torch.var(torch.tensor(self.loss_history['equity'][-window_size:]))
        var_eng = torch.var(torch.tensor(self.loss_history['engagement'][-window_size:]))
        
        variance_vector = torch.tensor([var_acc, var_eq, var_eng])
        
        self.weights = self.weights + self.meta_lr * variance_vector
        
        self.weights = self._project_simplex(self.weights)
        
        self.weights = torch.clamp(self.weights, min=self.lower_bound)
        self.weights = self.weights / self.weights.sum()
        
        logger.info(
            f"Epoch {epoch}: Updated weights to "
            f"[acc={self.weights[0]:.3f}, eq={self.weights[1]:.3f}, "
            f"eng={self.weights[2]:.3f}]"
        )
    
    def _project_simplex(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Project weights onto probability simplex.
        
        Ensures sum(weights) = 1 and all weights >= 0.
        
        Args:
            weights: Arbitrary weight vector
            
        Returns:
            Projected weights satisfying simplex constraints
        """
        sorted_weights, _ = torch.sort(weights, descending=True)
        cumsum = torch.cumsum(sorted_weights, dim=0)
        
        rho = (sorted_weights * torch.arange(1, len(weights) + 1) > (cumsum - 1)).sum() - 1
        theta = (cumsum[rho] - 1) / (rho + 1)
        
        projected = torch.clamp(weights - theta, min=0)
        
        return projected
    
    def get_composite_loss(
        self,
        loss_accuracy: torch.Tensor,
        loss_equity: torch.Tensor,
        loss_engagement: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted composite loss.
        
        Args:
            loss_accuracy: Accuracy objective loss
            loss_equity: Equity objective loss
            loss_engagement: Engagement objective loss
            
        Returns:
            Scalar composite loss for backpropagation
        """
        composite = (
            self.weights[0] * loss_accuracy +
            self.weights[1] * loss_equity +
            self.weights[2] * loss_engagement
        )
        
        return composite