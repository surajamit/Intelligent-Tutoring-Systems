#!/usr/bin/env python
"""
Generate all figures from the manuscript based on experimental results.

Author: Amit Pimpalkar
Organization: RBU, Nagpur
Year: 2026

This script loads the result CSVs (or dummy data) and produces the plots shown
in Figures 4-7 and supplementary figures.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.analysis.visualization import generate_all_figures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate figures from results')
    parser.add_argument('--result_dir', type=str, default='outputs/analytics',
                        help='Directory containing result CSV files')
    parser.add_argument('--output_dir', type=str, default='outputs/figures',
                        help='Directory to save figures')
    parser.add_argument('--dummy', action='store_true',
                        help='Generate dummy data for demonstration (if no real results)')
    return parser.parse_args()


def create_dummy_data() -> dict:
    """Create dummy data matching the tables in the manuscript."""
    datasets = ['EdNet', 'ASSISTments', 'OULAD']
    baseline_models = ['Decision Trees', 'Gamification', 'ML']
    
    # Table 1: Accuracy
    acc_data = {
        'Dataset': datasets,
        'Decision Trees': [86.7, 84.1, 82.5],
        'Gamification': [89.4, 87.6, 85.2],
        'ML': [91.2, 89.3, 87.9],
        'Proposed Model': [94.8, 93.1, 92.4]
    }
    acc_df = pd.DataFrame(acc_data)
    
    # Table 2: Disparity
    disp_data = {
        'Dataset': datasets,
        'Decision Trees': [11.8, 12.4, 13.1],
        'Gamification': [9.6, 10.1, 10.8],
        'ML': [8.3, 8.9, 9.5],
        'Proposed Model': [3.9, 4.2, 4.6]
    }
    disp_df = pd.DataFrame(disp_data)
    
    # Table 3: Engagement Retention
    eng_data = {
        'Dataset': datasets,
        'Decision Trees': [71.3, 69.5, 66.8],
        'Gamification': [83.9, 81.2, 79.6],
        'ML': [78.4, 76.8, 75.1],
        'Proposed Model': [89.6, 88.1, 86.9]
    }
    eng_df = pd.DataFrame(eng_data)
    
    # Table 4: Equity Stability (lower better)
    eq_data = {
        'Dataset': datasets,
        'Decision Trees': [0.42, 0.45, 0.48],
        'Gamification': [0.36, 0.39, 0.41],
        'ML': [0.31, 0.34, 0.37],
        'Proposed Model': [0.18, 0.20, 0.22]
    }
    eq_df = pd.DataFrame(eq_data)
    
    # Table 5: Counterfactual Divergence
    div_data = {
        'Dataset': datasets,
        'Decision Trees': [0.34, 0.37, 0.39],
        'Gamification': [0.29, 0.31, 0.33],
        'ML': [0.25, 0.27, 0.29],
        'Proposed Model': [0.11, 0.13, 0.15]
    }
    div_df = pd.DataFrame(div_data)
    
    # Table 6: Attention Alignment
    att_data = {
        'Dataset': datasets,
        'Decision Trees': [0.61, 0.59, 0.57],
        'Gamification': [0.68, 0.66, 0.64],
        'ML': [0.71, 0.69, 0.67],
        'Proposed Model': [0.87, 0.85, 0.83]
    }
    att_df = pd.DataFrame(att_data)
    
    # Table 7: Pareto Balance
    par_data = {
        'Dataset': datasets,
        'Decision Trees': [0.52, 0.50, 0.48],
        'Gamification': [0.63, 0.61, 0.59],
        'ML': [0.68, 0.66, 0.65],
        'Proposed Model': [0.89, 0.87, 0.85]
    }
    par_df = pd.DataFrame(par_data)
    
    # Table 8: Policy Retention
    ret_data = {
        'Dataset': datasets,
        'Decision Trees': [0.72, 0.70, 0.68],
        'Gamification': [0.81, 0.79, 0.77],
        'ML': [0.85, 0.83, 0.82],
        'Proposed Model': [0.96, 0.95, 0.94]
    }
    ret_df = pd.DataFrame(ret_data)
    
    # Table 9: Composite
    comp_data = {
        'Dataset': datasets,
        'Decision Trees': [0.68, 0.65, 0.63],
        'Gamification': [0.75, 0.73, 0.71],
        'ML': [0.78, 0.76, 0.75],
        'Proposed Model': [0.92, 0.90, 0.88]
    }
    comp_df = pd.DataFrame(comp_data)
    
    return {
        'accuracy': acc_df,
        'disparity': disp_df,
        'engagement': eng_df,
        'equity': eq_df,
        'divergence': div_df,
        'attention': att_df,
        'pareto': par_df,
        'retention': ret_df,
        'composite': comp_df
    }


def main():
    args = parse_args()
    
    if args.dummy:
        logger.info("Using dummy data for demonstration.")
        table_data = create_dummy_data()
    else:
        # Load from CSV files in result_dir
        result_dir = Path(args.result_dir)
        table_data = {}
        for table_name in ['accuracy', 'disparity', 'engagement', 'equity',
                           'divergence', 'attention', 'pareto', 'retention', 'composite']:
            csv_path = result_dir / f'{table_name}.csv'
            if csv_path.exists():
                table_data[table_name] = pd.read_csv(csv_path)
                logger.info(f"Loaded {csv_path}")
            else:
                logger.warning(f"CSV not found: {csv_path}. Skipping.")
    
    if not table_data:
        logger.error("No data found. Use --dummy to generate demo figures.")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    generate_all_figures(Path(args.result_dir), output_dir, table_data)
    logger.info("Figure generation complete.")


if __name__ == '__main__':
    main()