"""
Integrated Multi-Objective Tutoring System.

Author: Amit Pimpalkar
Organization: RBU, Nagpur
Year: 2026

Complete end-to-end architecture orchestrating all modules.
"""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dsge_module import DemographicSensitivityGradientEncoder
from .cern_module import CounterfactualEquityReplayNetwork
from .ewfaf_module import EngagementWeightedFairnessAttention
from .pareto_controller import ParetoAdaptiveObjectiveController

logger = logging.getLogger(__name__)


class IntegratedMultiObjectiveTutor(nn.Module):
    """
    Complete tutoring system architecture.
    
    Pipeline:
    1. Sequence encoding via GRU
    2. Demographic embedding
    3. Base mastery prediction
    4. DSGE computation
    5. CERN counterfactual replay
    6. EWFAF attention fusion
    7. Final classification
    """
    
    def __init__(self, config: Dict):
        """
        Initialize integrated architecture.
        
        Args:
            config: Configuration dictionary with model hyperparameters
        """
        super().__init__()
        
        self.hidden_dim = config['model']['hidden_dimension']
        self.demo_dim = config['model']['demographic_embedding_dim']
        
        self.encoder = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=config['model']['num_gru_layers'],
            batch_first=True,
            dropout=0.1 if config['model']['num_gru_layers'] > 1 else 0
        )
        
        self.demographic_embedding = nn.Embedding(
            num_embeddings=100,
            embedding_dim=self.demo_dim
        )
        
        self.dsge = DemographicSensitivityGradientEncoder(
            smoothing_factor=config['multi_objective']['gradient_smoothing_factor']
        )
        
        self.cern = CounterfactualEquityReplayNetwork(
            num_counterfactual_samples=config['multi_objective']['counterfactual_samples_per_trajectory']
        )
        
        self.ewfaf = EngagementWeightedFairnessAttention(
            hidden_dim=self.hidden_dim,
            num_attention_heads=config['model']['attention_heads'],
            attention_dropout=config['model']['attention_dropout']
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config['model']['classifier_dropout']),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"Initialized IntegratedMultiObjectiveTutor")
    
    def forward(
        self,
        states: torch.Tensor,
        demographic_ids: torch.Tensor,
        engagement: torch.Tensor,
        mask: torch.Tensor = None,
        demographic_pool: torch.Tensor = None,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete architecture.
        
        Args:
            states: [batch, seq_len, state_dim] interaction sequences
            demographic_ids: [batch] demographic group identifiers
            engagement: [batch, seq_len] engagement proxy signals
            mask: [batch, seq_len] valid position mask
            demographic_pool: [pool_size, demo_dim] for CERN sampling
            return_components: Whether to return intermediate outputs
            
        Returns:
            Dictionary containing predictions and optional component outputs
        """
        demo_embeddings = self.demographic_embedding(demographic_ids)
        demo_embeddings.requires_grad_(True)
        
        encoded, _ = self.encoder(states)
        
        base_predictions = self.classifier(encoded)
        
        base_loss = F.binary_cross_entropy(
            base_predictions,
            torch.zeros_like(base_predictions),
            reduction='mean'
        )
        dis_signal, gradients = self.dsge(base_loss, demo_embeddings)
        
        if demographic_pool is not None:
            cf_predictions, divergence = self.cern(
                base_model=lambda s, d: self.encoder(s)[0],
                states=states,
                factual_demographics=demo_embeddings,
                demographic_pool=demographic_pool
            )
        else:
            divergence = torch.tensor(0.0, device=states.device)
        
        equity_residuals = divergence.unsqueeze(0).expand(encoded.size(0), encoded.size(1))
        
        fused_representation = self.ewfaf(
            hidden_states=encoded,
            engagement_scores=engagement,
            equity_residuals=equity_residuals,
            attention_mask=mask
        )
        
        final_predictions = self.classifier(fused_representation)
        
        outputs = {
            'predictions': final_predictions,
            'dis_signal': dis_signal,
            'divergence': divergence
        }
        
        if return_components:
            outputs.update({
                'base_predictions': base_predictions,
                'encoded': encoded,
                'fused': fused_representation,
                'demo_embeddings': demo_embeddings
            })
        
        return outputs