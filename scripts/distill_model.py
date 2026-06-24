#!/usr/bin/env python
"""
Equity-Preserving Policy Distillation (EPPDV) as per Equation 13-14.

Author: Amit Pimpalkar
Organization: RBU, Nagpur
Year: 2026

Compresses the trained multi-objective teacher model into a lightweight student
network while maintaining equity retention ratio >= 0.95.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import IntegratedMultiObjectiveTutor, EquityPreservingDistillation
from src.data import load_educational_dataset
from src.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Run equity-preserving distillation')
    parser.add_argument('--teacher', type=str, required=True, help='Path to teacher checkpoint')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml')
    parser.add_argument('--dataset', type=str, required=True, choices=['ednet', 'assistments', 'oulad'])
    parser.add_argument('--output_dir', type=str, default='outputs/distilled')
    parser.add_argument('--epochs', type=int, default=30, help='Distillation epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Student learning rate')
    parser.add_argument('--compression_ratio', type=float, default=0.6, help='Target compression')
    return parser.parse_args()


def create_student_model(teacher_model: IntegratedMultiObjectiveTutor, compression_ratio: float) -> nn.Module:
    """
    Create a lightweight student network with fewer parameters.
    Halves hidden dimensions and reduces layers.
    """
    student_config = {
        'model': {
            'hidden_dimension': int(teacher_model.hidden_dim * compression_ratio),
            'demographic_embedding_dim': int(teacher_model.demo_dim * compression_ratio),
            'num_gru_layers': max(1, teacher_model.encoder.num_layers - 1),
            'attention_heads': max(2, teacher_model.ewfaf.num_heads - 2),
            'attention_dropout': 0.1,
            'classifier_dropout': 0.2
        }
    }
    # Create student with reduced dimensions
    student = IntegratedMultiObjectiveTutor(student_config)
    return student


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load configs
    config = yaml.safe_load(open(args.config))
    dataset_config = yaml.safe_load(open('configs/dataset_paths.yaml'))
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = load_educational_dataset(args.dataset, dataset_config, split='train')
    val_dataset = load_educational_dataset(args.dataset, dataset_config, split='val')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Load teacher
    logger.info(f"Loading teacher from {args.teacher}")
    teacher = IntegratedMultiObjectiveTutor(config).to(device)
    checkpoint = torch.load(args.teacher, map_location=device)
    teacher.load_state_dict(checkpoint['model_state_dict'])
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    # Create student
    logger.info(f"Creating student with compression ratio {args.compression_ratio}")
    student = create_student_model(teacher, args.compression_ratio).to(device)
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    logger.info(f"Teacher params: {teacher_params:,}, Student params: {student_params:,}")
    logger.info(f"Compression: {student_params / teacher_params:.2%}")
    
    # Optimizer
    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    distill_module = EquityPreservingDistillation(
        temperature=config['multi_objective']['distillation_temperature'],
        equity_retention_penalty=config['multi_objective']['equity_retention_penalty']
    )
    
    # Tracking for retention ratio
    best_retention = 0.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting distillation...")
    for epoch in range(args.epochs):
        student.train()
        total_distill_loss = 0.0
        
        for batch in train_loader:
            states = batch['states'].to(device)
            demographics = batch['demographics'].to(device)
            engagement = batch['engagement'].to(device)
            mask = batch['mask'].to(device)
            
            optimizer.zero_grad()
            
            # Teacher outputs (no grad)
            with torch.no_grad():
                teacher_outputs = teacher(
                    states=states,
                    demographic_ids=demographics.long(),
                    engagement=engagement,
                    mask=mask,
                    return_components=False
                )
                teacher_logits = teacher_outputs['predictions']
                teacher_equity = teacher_outputs['dis_signal'] * teacher_outputs['divergence']
            
            # Student outputs
            student_outputs = student(
                states=states,
                demographic_ids=demographics.long(),
                engagement=engagement,
                mask=mask,
                return_components=False
            )
            student_logits = student_outputs['predictions']
            student_equity = student_outputs['dis_signal'] * student_outputs['divergence']
            
            # Distillation loss
            loss = distill_module(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                student_equity_loss=student_equity,
                teacher_equity_loss=teacher_equity
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            
            total_distill_loss += loss.item()
        
        # Validation
        student.eval()
        val_accuracies = []
        with torch.no_grad():
            for batch in val_loader:
                states = batch['states'].to(device)
                demographics = batch['demographics'].to(device)
                engagement = batch['engagement'].to(device)
                targets = batch['targets'].to(device)
                mask = batch['mask'].to(device)
                
                outputs = student(
                    states=states,
                    demographic_ids=demographics.long(),
                    engagement=engagement,
                    mask=mask
                )
                preds = outputs['predictions'].squeeze(-1)
                valid_mask = mask.bool()
                acc = ((preds[valid_mask] > 0.5).float() == targets[valid_mask]).float().mean().item()
                val_accuracies.append(acc)
        
        val_acc = np.mean(val_accuracies)
        logger.info(f"Epoch {epoch+1}: Loss={total_distill_loss:.4f}, Val Acc={val_acc:.4f}")
    
    # Final evaluation: Compute Equity Retention Ratio
    logger.info("Final equity retention validation...")
    student.eval()
    teacher.eval()
    
    all_student_equity = []
    all_teacher_equity = []
    
    with torch.no_grad():
        for batch in val_loader:
            states = batch['states'].to(device)
            demographics = batch['demographics'].to(device)
            engagement = batch['engagement'].to(device)
            mask = batch['mask'].to(device)
            
            s_out = student(states, demographics.long(), engagement, mask)
            t_out = teacher(states, demographics.long(), engagement, mask)
            
            s_eq = (s_out['dis_signal'] * s_out['divergence']).item()
            t_eq = (t_out['dis_signal'] * t_out['divergence']).item()
            all_student_equity.append(s_eq)
            all_teacher_equity.append(t_eq)
    
    avg_student_eq = np.mean(all_student_equity)
    avg_teacher_eq = np.mean(all_teacher_equity)
    retention = distill_module.compute_equity_retention_ratio(avg_student_eq, avg_teacher_eq)
    
    logger.info(f"\n{'='*70}")
    logger.info("Distillation Complete")
    logger.info(f"Teacher Equity Loss: {avg_teacher_eq:.4f}")
    logger.info(f"Student Equity Loss: {avg_student_eq:.4f}")
    logger.info(f"Equity Retention Ratio: {retention:.4f} (Target: >= 0.95)")
    logger.info(f"{'='*70}")
    
    # Save student policy
    save_path = output_dir / 'student_policy.pth'
    torch.save({
        'model_state_dict': student.state_dict(),
        'config': config,
        'retention_ratio': retention,
        'teacher_equity': avg_teacher_eq,
        'student_equity': avg_student_eq
    }, save_path)
    logger.info(f"Student policy saved to {save_path}")


if __name__ == '__main__':
    main()