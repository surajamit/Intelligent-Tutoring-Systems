"""
Equity-Preserving Policy Distillation module.

Author: Amit Pimpalkar
Organization: RBU, Nagpur
Year: 2026

Compresses the full multi-objective teacher model into a lightweight student
network while maintaining equity guarantees.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class EquityPreservingDistillation(nn.Module):
    """
    Knowledge distillation with explicit equity retention constraint.
    
    Standard distillation minimizes KL divergence between teacher and student
    outputs. This approach augments the objective with an equity preservation
    term to ensure compressed models don't degrade fairness properties.
    """
    
    def __init__(
        self,
        temperature: float = 2.0,
        equity_retention_penalty: float = 0.5
    ):
        """
        Initialize distillation module.
        
        Args:
            temperature: Softening temperature for probability distributions
            equity_retention_penalty: Weight for equity preservation loss
        """
        super().__init__()
        self.temperature = temperature
        self.equity_penalty = equity_retention_penalty
        
        logger.info(
            f"Initialized distillation with T={temperature}, "
            f"penalty={equity_retention_penalty}"
        )
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_equity_loss: torch.Tensor,
        teacher_equity_loss: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distillation loss with equity preservation.
        
        Args:
            student_logits: [batch, seq_len, num_classes] student predictions
            teacher_logits: [batch, seq_len, num_classes] teacher predictions
            student_equity_loss: Equity loss from student forward pass
            teacher_equity_loss: Reference equity loss from teacher
            
        Returns:
            Combined distillation loss
        """
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_predictions = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        kl_loss = F.kl_div(
            soft_predictions,
            soft_targets,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        equity_degradation = F.relu(student_equity_loss - teacher_equity_loss)
        
        total_loss = kl_loss + self.equity_penalty * equity_degradation
        
        return total_loss
    
    def compute_equity_retention_ratio(
        self,
        student_equity: float,
        teacher_equity: float
    ) -> float:
        """
        Calculate equity retention ratio.
        
        Args:
            student_equity: Student model equity loss
            teacher_equity: Teacher model equity loss
            
        Returns:
            Retention ratio (target >= 0.95)
        """
        if teacher_equity == 0:
            return 1.0
        
        retention = 1.0 - (abs(student_equity - teacher_equity) / teacher_equity)
        
        return max(0.0, retention)