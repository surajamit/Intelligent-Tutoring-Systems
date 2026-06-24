"""
Counterfactual Equity Replay Networks (CERN) module.

Author: Amit Pimpalkar
Organization: RBU, Nagpur
Year: 2026

Performs demographic interventions to compute trajectory-level counterfactual
learning paths and measures divergence from factual outcomes.
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CounterfactualEquityReplayNetwork(nn.Module):
    """
    Simulates alternative demographic contexts over factual interaction sequences.
    
    For each trajectory:
    1. Samples K alternative demographic embeddings from empirical distribution
    2. Executes forward pass with intervened demographics
    3. Computes divergence between factual and counterfactual predictions
    4. Weights divergence by DIS to prioritize high-risk trajectories
    """
    
    def __init__(
        self,
        num_counterfactual_samples: int = 3,
        divergence_metric: str = 'l1'
    ):
        """
        Initialize CERN module.
        
        Args:
            num_counterfactual_samples: Number of alternative demographics per trajectory
            divergence_metric: Distance measure ['l1', 'l2', 'kl']
        """
        super().__init__()
        self.k_samples = num_counterfactual_samples
        self.divergence_metric = divergence_metric
        
        logger.info(
            f"Initialized CERN with k={num_counterfactual_samples}, "
            f"metric={divergence_metric}"
        )
    
    def forward(
        self,
        base_model: nn.Module,
        states: torch.Tensor,
        factual_demographics: torch.Tensor,
        demographic_pool: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute counterfactual trajectory divergence.
        
        Args:
            base_model: Shared encoder network (GRU/Transformer)
            states: [batch, seq_len, state_dim] interaction sequences
            factual_demographics: [batch, demo_dim] original demographics
            demographic_pool: [pool_size, demo_dim] empirical demographic samples
            
        Returns:
            Tuple of (counterfactual_predictions, divergence_residual)
        """
        batch_size = states.size(0)
        device = states.device
        
        with torch.no_grad():
            factual_output = base_model(states, factual_demographics)
        
        counterfactual_outputs = []
        
        for k in range(self.k_samples):
            cf_indices = torch.randint(
                0,
                demographic_pool.size(0),
                (batch_size,),
                device=device
            )
            cf_demographics = demographic_pool[cf_indices]
            
            cf_output = base_model(states, cf_demographics)
            counterfactual_outputs.append(cf_output)
        
        cf_stack = torch.stack(counterfactual_outputs, dim=1)
        
        if self.divergence_metric == 'l1':
            divergence = torch.abs(
                factual_output.unsqueeze(1) - cf_stack
            ).mean(dim=(1, 2, 3))
        elif self.divergence_metric == 'l2':
            divergence = torch.norm(
                factual_output.unsqueeze(1) - cf_stack,
                p=2,
                dim=-1
            ).mean(dim=(1, 2))
        else:
            factual_probs = torch.softmax(factual_output, dim=-1)
            cf_probs = torch.softmax(cf_stack, dim=-1)
            divergence = torch.nn.functional.kl_div(
                cf_probs.log(),
                factual_probs.unsqueeze(1),
                reduction='none'
            ).sum(dim=-1).mean(dim=(1, 2))
        
        mean_divergence = divergence.mean()
        
        mean_cf_output = cf_stack.mean(dim=1)
        
        return mean_cf_output, mean_divergence
    
    def compute_equity_loss(
        self,
        divergence: torch.Tensor,
        dis_signal: torch.Tensor,
        temporal_regularizer: float = 0.1
    ) -> torch.Tensor:
        """
        Construct equity loss from divergence and DIS.
        
        Args:
            divergence: Counterfactual trajectory divergence
            dis_signal: Demographic inequity signal from DSGE
            temporal_regularizer: Coefficient for divergence rate penalty
            
        Returns:
            Scalar equity loss for backpropagation
        """
        weighted_divergence = dis_signal * divergence
        
        equity_loss = weighted_divergence + temporal_regularizer * divergence
        
        return equity_loss