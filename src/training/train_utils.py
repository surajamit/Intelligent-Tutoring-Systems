"""
Core training loop implementation.

Author: Amit Pimpalkar
Organization: RBU, Nagpur
Year: 2026
"""

import logging
from typing import Dict

import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    pareto_controller,
    device,
    epoch: int,
    config: Dict
) -> Dict[str, float]:
    """
    Execute one training epoch.
    
    Returns:
        Dictionary of training metrics
    """
    model.train()
    
    total_loss = 0.0
    total_accuracy = 0.0
    total_equity = 0.0
    total_engagement = 0.0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        states = batch['states'].to(device)
        demographics = batch['demographics'].to(device)
        engagement = batch['engagement'].to(device)
        targets = batch['targets'].to(device)
        mask = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            states=states,
            demographic_ids=demographics.long(),
            engagement=engagement,
            mask=mask
        )
        
        predictions = outputs['predictions']
        accuracy_loss = nn.BCELoss()(predictions, targets.unsqueeze(-1))
        equity_loss = outputs['dis_signal'] * outputs['divergence']
        engagement_loss = (1.0 - engagement).mean()
        
        composite_loss = pareto_controller.get_composite_loss(
            accuracy_loss,
            equity_loss,
            engagement_loss
        )
        
        composite_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config['optimization']['gradient_clip_norm']
        )
        optimizer.step()
        
        total_loss += composite_loss.item()
        total_accuracy += accuracy_loss.item()
        total_equity += equity_loss.item()
        total_engagement += engagement_loss.item()
        
        progress_bar.set_postfix({
            'loss': composite_loss.item(),
            'acc': accuracy_loss.item()
        })
    
    pareto_controller.update_weights(
        total_accuracy / len(dataloader),
        total_equity / len(dataloader),
        total_engagement / len(dataloader),
        epoch
    )
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': total_accuracy / len(dataloader),
        'equity_loss': total_equity / len(dataloader),
        'engagement_loss': total_engagement / len(dataloader)
    }


def validate(model: nn.Module, dataloader, device) -> Dict[str, float]:
    """
    Validation procedure.
    """
    model.eval()
    
    total_accuracy = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            states = batch['states'].to(device)
            demographics = batch['demographics'].to(device)
            engagement = batch['engagement'].to(device)
            targets = batch['targets'].to(device)
            mask = batch['mask'].to(device)
            
            outputs = model(
                states=states,
                demographic_ids=demographics.long(),
                engagement=engagement,
                mask=mask
            )
            
            predictions = outputs['predictions']
            accuracy = ((predictions > 0.5).float() == targets.unsqueeze(-1)).float().mean()
            total_accuracy += accuracy.item()
    
    return {
        'accuracy': total_accuracy / len(dataloader),
        'composite_score': total_accuracy / len(dataloader)
    }