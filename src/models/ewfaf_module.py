"""
Engagement-Weighted Fairness Attention Fusion (EWFAF) module.

Author: Amit Pimpalkar
Organization: RBU, Nagpur
Year: 2026

Conditions attention mechanisms on residual equity risk to prevent engagement
optimization from overriding fairness in high-disparity regions.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class EngagementWeightedFairnessAttention(nn.Module):
    """
    Multi-head attention conditioned on engagement and equity signals.
    
    Standard attention mechanisms allocate focus based on activation magnitudes,
    which often correlate with high-engagement (advantaged) learners. This module
    gates attention using fairness residuals to redistribute focus toward learners
    experiencing structural disadvantages.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
        fairness_gating_threshold: float = 0.01
    ):
        """
        Initialize EWFAF attention layer.
        
        Args:
            hidden_dim: Dimension of hidden state vectors
            num_attention_heads: Number of parallel attention heads
            attention_dropout: Dropout probability for attention weights
            fairness_gating_threshold: Minimum residual to activate gating
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_attention_heads
        self.fairness_threshold = fairness_gating_threshold
        
        self.engagement_projection = nn.Linear(1, hidden_dim)
        self.fairness_projection = nn.Linear(1, hidden_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        logger.info(
            f"Initialized EWFAF with {num_heads} heads, "
            f"threshold={fairness_gating_threshold}"
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        engagement_scores: torch.Tensor,
        equity_residuals: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Apply fairness-gated attention fusion.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim] encoder outputs
            engagement_scores: [batch, seq_len] persistence/latency signals
            equity_residuals: [batch, seq_len] counterfactual divergence per timestep
            attention_mask: [batch, seq_len] valid position indicators
            
        Returns:
            Fused representation tensor [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        engagement_proj = self.engagement_projection(
            engagement_scores.unsqueeze(-1)
        )
        
        fairness_proj = self.fairness_projection(
            equity_residuals.unsqueeze(-1)
        )
        
        query = hidden_states + engagement_proj
        
        key_value = hidden_states + fairness_proj
        
        attended_output, attention_weights = self.attention(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
        )
        
        attended_output = self.layer_norm(hidden_states + attended_output)
        
        gate_input = torch.cat([engagement_proj, fairness_proj], dim=-1)
        fusion_weights = self.fusion_gate(gate_input)
        
        gated_output = fusion_weights * attended_output + (1 - fusion_weights) * hidden_states
        
        return gated_output
    
    def compute_attention_fairness_loss(
        self,
        attention_weights: torch.Tensor,
        equity_residuals: torch.Tensor
    ) -> torch.Tensor:
        """
        Penalize attention concentration in low-equity regions.
        
        Args:
            attention_weights: [batch, num_heads, seq_len, seq_len]
            equity_residuals: [batch, seq_len]
            
        Returns:
            Scalar regularization loss
        """
        avg_attention = attention_weights.mean(dim=1)
        
        attention_entropy = -(avg_attention * torch.log(avg_attention + 1e-9)).sum(dim=-1)
        
        high_risk_mask = (equity_residuals > self.fairness_threshold).float()
        
        fairness_penalty = (high_risk_mask * (1.0 - attention_entropy)).mean()
        
        return fairness_penalty