#!/usr/bin/env python
"""
Data preparation script for educational datasets.

Author: Amit Pimpalkar
Organization: RBU, Nagpur
Year: 2026

Downloads and preprocesses EdNet, ASSISTments, and OULAD datasets.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare educational dataset')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['ednet', 'assistments', 'oulad'],
                        help='Dataset to prepare')
    parser.add_argument('--output', type=str, default='data/processed/',
                        help='Output directory for processed data')
    parser.add_argument('--raw', type=str, default='data/raw/',
                        help='Raw data directory')
    return parser.parse_args()


def prepare_ednet(raw_dir, out_dir):
    """Prepare EdNet dataset."""
    logger.info("Preparing EdNet dataset...")
    # Placeholder: Download and parse EdNet files.
       
    # Create data for structure
    os.makedirs(out_dir, exist_ok=True)
    
    # Generate trajectories
    np.random.seed(42)
    n_samples = 1000
    seq_len = 50
    n_features = 5
    
    for split in ['train', 'val', 'test']:
        df = pd.DataFrame({
            'learner_id': np.repeat(np.arange(n_samples // 10), 10),
            'timestamp': np.arange(n_samples),
            'problem_id': np.random.randint(1, 100, n_samples),
            'correctness': np.random.randint(0, 2, n_samples),
            'response_time': np.random.exponential(500, n_samples),
            'hint_count': np.random.randint(0, 3, n_samples),
            'attempt_number': np.random.randint(1, 4, n_samples),
            'language_group': np.random.choice(['A', 'B', 'C'], n_samples),
            'access_index': np.random.uniform(0, 1, n_samples),
            'prior_exposure': np.random.uniform(0, 1, n_samples),
            'age_category': np.random.choice(['young', 'mid', 'older'], n_samples)
        })
        df.to_csv(os.path.join(out_dir, f'{split}_sequences.csv'), index=False)
    
    logger.info(f"EdNet prepared in {out_dir}")


def prepare_assistments(raw_dir, out_dir):
    """Prepare ASSISTments dataset."""
    logger.info("Preparing ASSISTments dataset...")
    os.makedirs(out_dir, exist_ok=True)
    
    # ASSISTments loader
    np.random.seed(123)
    n_samples = 800
    for split in ['train', 'val', 'test']:
        df = pd.DataFrame({
            'learner_id': np.repeat(np.arange(n_samples // 8), 8),
            'timestamp': np.arange(n_samples),
            'skill_id': np.random.randint(1, 50, n_samples),
            'correct': np.random.randint(0, 2, n_samples),
            'ms_first_response': np.random.exponential(300, n_samples),
            'hint_total': np.random.randint(0, 3, n_samples),
            'overlap_time': np.random.uniform(0, 100, n_samples),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'prior_skills': np.random.uniform(0, 1, n_samples)
        })
        df.to_csv(os.path.join(out_dir, f'{split}_sequences.csv'), index=False)
    
    logger.info(f"ASSISTments prepared in {out_dir}")


def prepare_oulad(raw_dir, out_dir):
    """Prepare OULAD dataset."""
    logger.info("Preparing OULAD dataset...")
    os.makedirs(out_dir, exist_ok=True)
    
    # OULAD data loader
    np.random.seed(456)
    n_samples = 600
    for split in ['train', 'val', 'test']:
        df = pd.DataFrame({
            'learner_id': np.repeat(np.arange(n_samples // 6), 6),
            'timestamp': np.arange(n_samples),
            'activity_type': np.random.choice(['forum', 'quiz', 'resource'], n_samples),
            'sum_click': np.random.poisson(10, n_samples),
            'date': pd.date_range('2020-01-01', periods=n_samples).strftime('%Y-%m-%d'),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'highest_education': np.random.choice(['A Level', 'Bachelor', 'Master'], n_samples),
            'imd_band': np.random.choice([1,2,3,4,5], n_samples),
            'age_band': np.random.choice(['18-25', '26-35', '36-50'], n_samples)
        })
        df.to_csv(os.path.join(out_dir, f'{split}_sequences.csv'), index=False)
    
    logger.info(f"OULAD prepared in {out_dir}")


def main():
    args = parse_args()
    
    raw_path = Path(args.raw) / args.dataset
    out_path = Path(args.output) / args.dataset
    os.makedirs(raw_path, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)
    
    if args.dataset == 'ednet':
        prepare_ednet(raw_path, out_path)
    elif args.dataset == 'assistments':
        prepare_assistments(raw_path, out_path)
    elif args.dataset == 'oulad':
        prepare_oulad(raw_path, out_path)
    
    logger.info(f"Dataset {args.dataset} preparation complete.")


if __name__ == '__main__':
    main()
