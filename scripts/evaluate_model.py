"""
Model evaluation script for equity-aware tutoring system.

Author: Amit Pimpalkar
Organization: RBU, Nagpur
Year: 2026

Performs comprehensive evaluation across all metrics.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_educational_dataset
from src.models import IntegratedMultiObjectiveTutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, required=True, choices=['ednet', 'assistments', 'oulad'])
    parser.add_argument('--config', type=str, default='configs/training_config.yaml')
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    config = yaml.safe_load(open(args.config))
    dataset_config = yaml.safe_load(open('configs/dataset_paths.yaml'))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load test data
    test_dataset = load_educational_dataset(args.dataset, dataset_config, split='test')
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Load model
    model = IntegratedMultiObjectiveTutor(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Loaded model from {args.checkpoint}")
    
    # Evaluation
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
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
            
            predictions = outputs['predictions'].squeeze(-1)
            
            # Only consider valid timesteps
            valid_mask = mask.bool()
            preds = predictions[valid_mask].cpu().numpy()
            targets_flat = targets[valid_mask].cpu().numpy()
            
            all_predictions.extend(preds)
            all_targets.extend(targets_flat)
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Compute metrics
    pred_binary = (all_predictions > 0.5).astype(int)
    
    accuracy = accuracy_score(all_targets, pred_binary)
    roc_auc = roc_auc_score(all_targets, all_predictions)
    cm = confusion_matrix(all_targets, pred_binary)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Evaluation Results on {args.dataset.upper()}")
    logger.info(f"{'='*70}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # Compute demographic group performance
    # (In full implementation, would load demographic labels and compute per-group metrics)
    logger.info(f"\nEvaluation complete.")


if __name__ == '__main__':
    main()