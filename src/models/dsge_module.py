"""
Demographic Sensitivity Gradient Encoding (DSGE) module.

Author: Amit Pimpalkar
Organization: RBU, Nagpur
Year: 2026

Quantifies gradient-level demographic influence on learning predictions.
Computes a scalar Demographic Inequity Signal (DIS) by integrating gradient
norms over the temporal trajectory.
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DemographicSensitivityGradientEncoder(nn.Module):
    """
    Computes instantaneous sensitivity of predictions to demographic embeddings.
    
    The module calculates:
    1. Gradient of learning loss with respect to demographic embeddings
    2. L2 norm of gradient vector (instantaneous sensitivity)
    3. Temporal integration with second-order smoothing
    
    This provides a continuous signal quantifying structural inequity that can be
    backpropagated through the network.
    """
    
    def __init__(
        self,
        smoothing_factor: float = 0.05,
        second_order_regularization: float = 0.01
    ):
        """
        Initialize DSGE module.
        
        Args:
            smoothing_factor: Weight for second-order gradient regularization
            second_order_regularization: Coefficient for curvature smoothing
        """
        super().__init__()
        self.smoothing_factor = smoothing_factor
        self.second_order_reg = second_order_regularization
        
        logger.info(
            f"Initialized DSGE with smoothing={smoothing_factor}, "
            f"second_order={second_order_regularization}"
        )
    
    def forward(
        self,
        base_loss: torch.Tensor,
        demographic_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute demographic sensitivity signal.
        
        Args:
            base_loss: Scalar loss from base learning prediction
            demographic_embeddings: Tensor [batch, demo_dim] requiring gradients
            
        Returns:
            Tuple of (demographic_inequity_signal, gradient_tensor)
            
        Raises:
            RuntimeError: If demographic_embeddings doesn't require gradients
        """
        if not demographic_embeddings.requires_grad:
            raise RuntimeError(
                "Demographic embeddings must have requires_grad=True for DSGE"
            )
        
        gradients = torch.autograd.grad(
            outputs=base_loss,
            inputs=demographic_embeddings,
            retain_graph=True,
            create_graph=True,
            allow_unused=False
        )[0]
        
        gradient_norms = torch.norm(gradients, p=2, dim=1)
        
        integrated_sensitivity = gradient_norms.mean()
        
        if self.second_order_reg > 0:
            gradient_variance = torch.var(gradients, dim=0).mean()
            stabilized_signal = (
                integrated_sensitivity +
                self.smoothing_factor * gradient_variance
            )
        else:
            stabilized_signal = integrated_sensitivity
        
        return stabilized_signal, gradients
    
    def compute_temporal_dis(
        self,
        loss_sequence: torch.Tensor,
        demographic_sequence: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute DIS integrated over temporal sequence.
        
        Args:
            loss_sequence: [batch, seq_len] temporal losses
            demographic_sequence: [batch, demo_dim] demographic embeddings
            
        Returns:
            Scalar demographic inequity signal
        """
        dis_values = []
        
        for t in range(loss_sequence.size(1)):
            timestep_loss = loss_sequence[:, t].mean()
            dis_t, _ = self.forward(timestep_loss, demographic_sequence)
            dis_values.append(dis_t)
        
        temporal_dis = torch.stack(dis_values).mean()
        
        return temporal_dis