"""
Main training script for equity-aware tutoring system.

Author: Amit Pimpalkar
Organization: RBU, Nagpur
Year: 2026

Orchestrates the complete training pipeline including data loading,
multi-objective optimization, and checkpointing.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_educational_dataset, EquityAwareBatchSampler
from src.models import IntegratedMultiObjectiveTutor, ParetoAdaptiveObjectiveController
from src.training import train_one_epoch, validate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Train equity-aware intelligent tutoring system'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training_config.yaml',
        help='Path to training configuration file'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['ednet', 'assistments', 'oulad'],
        help='Dataset to train on'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/experiments',
        help='Directory for outputs and checkpoints'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_output_directory(base_dir: str, experiment_name: str) -> Path:
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)
    
    return output_dir


def main():
    args = parse_arguments()
    
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    output_dir = setup_output_directory(args.output_dir, config['experiment']['name'])
    logger.info(f"Output directory: {output_dir}")
    
    torch.manual_seed(config['experiment']['seed'])
    
    device = torch.device(config['experiment']['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    with open('configs/dataset_paths.yaml', 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    logger.info(f"Loading {args.dataset} dataset...")
    train_dataset = load_educational_dataset(args.dataset, dataset_config, split='train')
    val_dataset = load_educational_dataset(args.dataset, dataset_config, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    model = IntegratedMultiObjectiveTutor(config).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['optimization']['base_learning_rate'],
        weight_decay=config['optimization']['weight_decay'],
        betas=(config['optimization']['beta1'], config['optimization']['beta2'])
    )
    
    pareto_controller = ParetoAdaptiveObjectiveController(
        meta_learning_rate=config['multi_objective']['meta_learning_rate'],
        weight_lower_bound=config['multi_objective']['weight_lower_bound'],
        update_frequency=config['multi_objective']['weight_update_frequency']
    )
    
    best_composite_score = 0.0
    
    for epoch in range(config['optimization']['epochs']):
        logger.info(f"\n{'='*70}")
        logger.info(f"Epoch {epoch + 1}/{config['optimization']['epochs']}")
        logger.info(f"{'='*70}")
        
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            pareto_controller=pareto_controller,
            device=device,
            epoch=epoch,
            config=config
        )
        
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            device=device
        )
        
        logger.info(f"Train - Acc: {train_metrics['accuracy']:.4f}, "
                   f"Equity: {train_metrics['equity_loss']:.4f}, "
                   f"Engagement: {train_metrics['engagement_loss']:.4f}")
        logger.info(f"Val   - Acc: {val_metrics['accuracy']:.4f}, "
                   f"Composite: {val_metrics['composite_score']:.4f}")
        
        if val_metrics['composite_score'] > best_composite_score:
            best_composite_score = val_metrics['composite_score']
            checkpoint_path = output_dir / 'checkpoints' / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, checkpoint_path)
            logger.info(f"Saved best model checkpoint: {checkpoint_path}")
    
    logger.info(f"\nTraining complete. Best composite score: {best_composite_score:.4f}")


if __name__ == '__main__':
    main()