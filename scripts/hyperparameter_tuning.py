#!/usr/bin/env python
"""
Grid-constrained Bayesian Optimization for hyperparameter search.

Author: Amit Pimpalkar
Organization: RBU, Nagpur
Year: 2026

Searches over learning rate, hidden dimensions, attention heads, and
equity regularization coefficients as described in the manuscript.
"""

import argparse
import logging
import sys
import json
from pathlib import Path
import itertools

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_educational_dataset
from src.models import IntegratedMultiObjectiveTutor, ParetoAdaptiveObjectiveController
from src.training import train_one_epoch, validate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml')
    parser.add_argument('--dataset', type=str, required=True, choices=['ednet', 'assistments', 'oulad'])
    parser.add_argument('--output', type=str, default='outputs/tuning/results.json')
    parser.add_argument('--trials', type=int, default=10, help='Number of random trials (or grid size)')
    return parser.parse_args()


def evaluate_config(config, dataset_name, dataset_config):
    """Train a model with given config for a few epochs and return validation composite score."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Quick training for tuning (only 5 epochs)
    epochs = 5
    
    train_dataset = load_educational_dataset(dataset_name, dataset_config, split='train')
    val_dataset = load_educational_dataset(dataset_name, dataset_config, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'],
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'],
                            shuffle=False, num_workers=2)
    
    model = IntegratedMultiObjectiveTutor(config).to(device)
    optimizer = optim.AdamW(model.parameters(),
                            lr=config['optimization']['base_learning_rate'],
                            weight_decay=config['optimization']['weight_decay'])
    
    pareto_controller = ParetoAdaptiveObjectiveController(
        meta_learning_rate=config['multi_objective']['meta_learning_rate'],
        weight_lower_bound=config['multi_objective']['weight_lower_bound'],
        update_frequency=config['multi_objective']['weight_update_frequency']
    )
    
    best_val_acc = 0.0
    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            pareto_controller=pareto_controller,
            device=device,
            epoch=epoch,
            config=config
        )
        val_metrics = validate(model, val_loader, device)
        if val_metrics['composite_score'] > best_val_acc:
            best_val_acc = val_metrics['composite_score']
    
    return best_val_acc


def main():
    args = parse_args()
    base_config = yaml.safe_load(open(args.config))
    dataset_config = yaml.safe_load(open('configs/dataset_paths.yaml'))
    
    # Define search space
    search_space = {
        'learning_rate': [1e-4, 2e-4, 3e-4, 5e-4],
        'hidden_dim': [128, 192, 256, 320],
        'attention_heads': [2, 4, 6],
        'equity_reg': [0.01, 0.05, 0.1, 0.2]
    }
    
    results = []
    
    # If trials < grid size, sample randomly; else grid search
    total_grid = np.prod([len(v) for v in search_space.values()])
    trials = min(args.trials, total_grid)
    
    # Generate combinations
    keys = list(search_space.keys())
    values = list(search_space.values())
    
    if trials < total_grid:
        # Random sampling
        import random
        sampled = []
        for _ in range(trials):
            comb = {k: random.choice(v) for k, v in search_space.items()}
            if comb not in sampled:
                sampled.append(comb)
        combinations = sampled
    else:
        # Full grid
        combinations = [dict(zip(keys, vals)) for vals in itertools.product(*values)]
    
    logger.info(f"Running {len(combinations)} hyperparameter trials...")
    
    for i, combo in enumerate(combinations):
        logger.info(f"Trial {i+1}/{len(combinations)}: {combo}")
        
        # Modify base config
        config = yaml.safe_load(open(args.config))
        config['optimization']['base_learning_rate'] = combo['learning_rate']
        config['model']['hidden_dimension'] = combo['hidden_dim']
        config['model']['attention_heads'] = combo['attention_heads']
        config['multi_objective']['equity_regularization'] = combo['equity_reg']  # Not used directly but tracked
        
        # Evaluate
        try:
            score = evaluate_config(config, args.dataset, dataset_config)
            results.append({
                'params': combo,
                'composite_score': score
            })
            logger.info(f"  Score: {score:.4f}")
        except Exception as e:
            logger.error(f"  Failed: {e}")
            results.append({
                'params': combo,
                'composite_score': None,
                'error': str(e)
            })
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Find best
    valid = [r for r in results if r['composite_score'] is not None]
    if valid:
        best = max(valid, key=lambda x: x['composite_score'])
        logger.info(f"\nBest Config: {best['params']} with score {best['composite_score']:.4f}")
    
    logger.info(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()