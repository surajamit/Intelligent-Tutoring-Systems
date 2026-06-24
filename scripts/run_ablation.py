#!/usr/bin/env python
"""
Ablation study script to evaluate the contribution of each component.

Author: Amit Pimpalkar
Organization: RBU, Nagpur
Year: 2026

Trains variants of the model with one component removed (e.g., no DSGE, no CERN,
no EWFAF, no Pareto adaptivity, no distillation) and compares performance.
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_educational_dataset
from src.models import IntegratedMultiObjectiveTutor, ParetoAdaptiveObjectiveController
from src.training import train_one_epoch, validate
from src.analysis.statistical_tests import paired_t_test, wilcoxon_signed_rank_test

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Run ablation study')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['ednet', 'assistments', 'oulad'])
    parser.add_argument('--output_dir', type=str, default='outputs/ablation')
    parser.add_argument('--epochs', type=int, default=20, help='Reduced epochs for ablation')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def create_ablation_model(config: dict, disable_components: list) -> IntegratedMultiObjectiveTutor:
    """
    Create model with certain components disabled.
    
    Args:
        config: Base configuration
        disable_components: List of strings: 'dsge', 'cern', 'ewfaf', 'pareto', 'distillation'
    
    Returns:
        Modified model 
    """
    # We'll modify the model class on the fly; for simplicity we create a wrapper.
    # In a full implementation, we would subclass IntegratedMultiObjectiveTutor
    # and override forward to skip selected modules.
    # Here we create a modified version by monkey-patching or using a flag.
    model = IntegratedMultiObjectiveTutor(config)
    
    # Store disabled list in model for conditional execution
    model.disabled = set(disable_components)
    
    # Save original forward
    original_forward = model.forward
    
    def new_forward(self, states, demographic_ids, engagement, mask=None,
                    demographic_pool=None, return_components=False):
        demo_embeddings = self.demographic_embedding(demographic_ids)
        demo_embeddings.requires_grad_(True)
        encoded, _ = self.encoder(states)
        
        # Base prediction without equity modules
        base_predictions = self.classifier(encoded)
        outputs = {}
        
        if 'dsge' in self.disabled:
            # Bypass DSGE: set dis_signal to zero
            dis_signal = torch.tensor(0.0, device=states.device)
            gradients = None
        else:
            base_loss = torch.nn.functional.binary_cross_entropy(
                base_predictions, torch.zeros_like(base_predictions), reduction='mean')
            dis_signal, gradients = self.dsge(base_loss, demo_embeddings)
        
        if 'cern' in self.disabled:
            divergence = torch.tensor(0.0, device=states.device)
        else:
            if demographic_pool is not None:
                _, divergence = self.cern(
                    base_model=lambda s, d: self.encoder(s)[0],
                    states=states,
                    factual_demographics=demo_embeddings,
                    demographic_pool=demographic_pool
                )
            else:
                divergence = torch.tensor(0.0, device=states.device)
        
        if 'ewfaf' in self.disabled:
            # Use simple concatenation or sum without attention
            fused_representation = encoded
        else:
            equity_residuals = divergence.unsqueeze(0).expand(encoded.size(0), encoded.size(1))
            fused_representation = self.ewfaf(
                hidden_states=encoded,
                engagement_scores=engagement,
                equity_residuals=equity_residuals,
                attention_mask=mask
            )
        
        final_predictions = self.classifier(fused_representation)
        
        outputs['predictions'] = final_predictions
        outputs['dis_signal'] = dis_signal if not 'dsge' in self.disabled else torch.tensor(0.0)
        outputs['divergence'] = divergence if not 'cern' in self.disabled else torch.tensor(0.0)
        
        if return_components:
            outputs.update({
                'base_predictions': base_predictions,
                'encoded': encoded,
                'fused': fused_representation,
                'demo_embeddings': demo_embeddings
            })
        return outputs
    
    # Apply new forward
    model.forward = new_forward.__get__(model, IntegratedMultiObjectiveTutor)
    return model


def run_ablation(config, dataset_name, output_dir, epochs, seed):
    """
    Run ablation for each component removal and collect results.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    
    # Load dataset config
    with open('configs/dataset_paths.yaml', 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    train_dataset = load_educational_dataset(dataset_name, dataset_config, split='train')
    val_dataset = load_educational_dataset(dataset_name, dataset_config, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'],
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'],
                            shuffle=False, num_workers=4)
    
    # Define ablation variants
    variants = {
        'full': [],
        'no_dsge': ['dsge'],
        'no_cern': ['cern'],
        'no_ewfaf': ['ewfaf'],
        'no_pareto': ['pareto'],  # will use fixed weights
        'no_distillation': ['distillation']  # not used in training, just for final evaluation
    }
    # We'll also add a variant without both dsge and cern 
    
    results = {}
    
    for variant_name, disable_list in variants.items():
        logger.info(f"Running ablation: {variant_name}")
        
        # Create model with disabled components
        model = create_ablation_model(config, disable_list)
        model.to(device)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=config['optimization']['base_learning_rate'],
                                      weight_decay=config['optimization']['weight_decay'])
        
        # Pareto controller: if disabled, use fixed weights
        if 'pareto' in disable_list:
            # Use fixed weights (initial)
            pareto_controller = ParetoAdaptiveObjectiveController(
                meta_learning_rate=0.0,  # no updates
                weight_lower_bound=0.05,
                update_frequency=1000  # never update
            )
        else:
            pareto_controller = ParetoAdaptiveObjectiveController(
                meta_learning_rate=config['multi_objective']['meta_learning_rate'],
                weight_lower_bound=config['multi_objective']['weight_lower_bound'],
                update_frequency=config['multi_objective']['weight_update_frequency']
            )
        
        # Train for reduced epochs
        best_val_acc = 0.0
        val_accs = []
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
            val_accs.append(val_metrics['accuracy'])
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
        
        # Record best validation accuracy and final train metrics
        results[variant_name] = {
            'best_val_accuracy': best_val_acc,
            'val_accuracies': val_accs,
            'final_train_accuracy': train_metrics['accuracy'],
            'final_equity_loss': train_metrics['equity_loss'],
            'final_engagement_loss': train_metrics['engagement_loss']
        }
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(results, orient='index')
    output_path = Path(output_dir) / f'ablation_{dataset_name}.csv'
    df.to_csv(output_path)
    logger.info(f"Ablation results saved to {output_path}")
    
    # Print summary
    logger.info("\nAblation Study Summary:")
    logger.info(df[['best_val_accuracy', 'final_train_accuracy']].to_string())
    
    # Statistical comparison: full vs each variant
    full_accs = np.array(results['full']['val_accuracies'])
    for variant, data in results.items():
        if variant == 'full':
            continue
        variant_accs = np.array(data['val_accuracies'])
        t_test = paired_t_test(full_accs, variant_accs)
        wilcox = wilcoxon_signed_rank_test(full_accs, variant_accs)
        logger.info(f"\n{variant} vs Full:")
        logger.info(f"  t-test: t={t_test['t_statistic']:.4f}, p={t_test['p_value']:.4f} (significant: {t_test['significant']})")
        logger.info(f"  Wilcoxon: W={wilcox['statistic']:.4f}, p={wilcox['p_value']:.4f} (significant: {wilcox['significant']})")
    
    return results


def main():
    args = parse_args()
    
    config = yaml.safe_load(open(args.config))
    config['optimization']['epochs'] = args.epochs  # override
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_ablation(config, args.dataset, output_dir, args.epochs, args.seed)


if __name__ == '__main__':
    main()
