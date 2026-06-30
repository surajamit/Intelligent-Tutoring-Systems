"""
Visualization functions.

Author: Amit Pimpalkar
Organization: RBU, Nagpur
Year: 2026

Plots for accuracy, disparity, engagement,
equity stability, counterfactual divergence, attention alignment,
Pareto balance, policy retention, and composite scores.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def set_style():
    """Set consistent plotting style."""
    sns.set_theme(style='whitegrid', context='paper')
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 9


def plot_accuracy_comparison(accuracy_df: pd.DataFrame, output_path: Optional[Path] = None):
    """
    Model's Integrated Accuracy Analysis (bar chart).

    Args:
        accuracy_df: DataFrame with columns ['Dataset', 'Decision Trees', 'Gamification', 'ML', 'Proposed Model']
        output_path: Where to save the figure (PNG/PDF)
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    datasets = accuracy_df['Dataset'].values
    x = np.arange(len(datasets))
    width = 0.2
    
    models = ['Decision Trees', 'Gamification', 'ML', 'Proposed Model']
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    
    for i, (model, color) in enumerate(zip(models, colors)):
        values = accuracy_df[model].values
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, values, width, label=model, color=color)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Learning Accuracy (%)')
    ax.set_title('Overall Learning Accuracy Across Contextual Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc='lower right')
    ax.set_ylim(80, 100)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_disparity_reduction(disparity_df: pd.DataFrame, output_path: Optional[Path] = None):
    """
    Model's Disparity Analysis (bar chart).
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    datasets = disparity_df['Dataset'].values
    models = ['Decision Trees', 'Gamification', 'ML', 'Proposed Model']
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, (model, color) in enumerate(zip(models, colors)):
        values = disparity_df[model].values
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, values, width, label=model, color=color)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Demographic Learning Gain Disparity (%)')
    ax.set_title('Demographic Learning Gain Disparity')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_engagement_retention(engagement_df: pd.DataFrame, output_path: Optional[Path] = None):
    """
    Model's Integrated Engagement Analysis.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    datasets = engagement_df['Dataset'].values
    models = ['Decision Trees', 'Gamification', 'ML', 'Proposed Model']
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, (model, color) in enumerate(zip(models, colors)):
        values = engagement_df[model].values
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, values, width, label=model, color=color)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Engagement Retention Rate (%)')
    ax.set_title('Engagement Retention Rate Over Sessions')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc='lower right')
    ax.set_ylim(60, 100)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_equity_stability(equity_df: pd.DataFrame, output_path: Optional[Path] = None):
    """
    Equity Stability Index (lower is better).
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    datasets = equity_df['Dataset'].values
    models = ['Decision Trees', 'Gamification', 'ML', 'Proposed Model']
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, (model, color) in enumerate(zip(models, colors)):
        values = equity_df[model].values
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, values, width, label=model, color=color)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Equity Stability Index (Lower is Better)')
    ax.set_title('Equity Stability Index')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_counterfactual_divergence(div_df: pd.DataFrame, output_path: Optional[Path] = None):
    """
    Counterfactual Trajectory Divergence.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    datasets = div_df['Dataset'].values
    models = ['Decision Trees', 'Gamification', 'ML', 'Proposed Model']
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, (model, color) in enumerate(zip(models, colors)):
        values = div_df[model].values
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, values, width, label=model, color=color)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Counterfactual Trajectory Divergence')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_attention_alignment(attention_df: pd.DataFrame, output_path: Optional[Path] = None):
    """
    Attention Fairness Alignment Score
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    datasets = attention_df['Dataset'].values
    models = ['Decision Trees', 'Gamification', 'ML', 'Proposed Model']
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, (model, color) in enumerate(zip(models, colors)):
        values = attention_df[model].values
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, values, width, label=model, color=color)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Attention Fairness Alignment Score')
    ax.set_title('Attention Fairness Alignment Score')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc='lower right')
    ax.set_ylim(0.5, 1.0)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pareto_balance(pareto_df: pd.DataFrame, output_path: Optional[Path] = None):
    """
    Pareto Optimization Balance Score
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    datasets = pareto_df['Dataset'].values
    models = ['Decision Trees', 'Gamification', 'ML', 'Proposed Model']
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, (model, color) in enumerate(zip(models, colors)):
        values = pareto_df[model].values
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, values, width, label=model, color=color)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Pareto Balance Score')
    ax.set_title('Pareto Optimization Balance Score')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc='lower right')
    ax.set_ylim(0.4, 1.0)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_policy_retention(retention_df: pd.DataFrame, output_path: Optional[Path] = None):
    """
    Policy Distillation Equity Retention Ratio
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    datasets = retention_df['Dataset'].values
    models = ['Decision Trees', 'Gamification', 'ML', 'Proposed Model']
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, (model, color) in enumerate(zip(models, colors)):
        values = retention_df[model].values
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, values, width, label=model, color=color)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Equity Retention Ratio')
    ax.set_title('Policy Distillation Equity Retention Ratio')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc='lower right')
    ax.set_ylim(0.6, 1.0)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_composite_scores(composite_df: pd.DataFrame, output_path: Optional[Path] = None):
    """
    Composite Equity-Accuracy-Engagement Score
    Model's Overall Result Analysis.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    datasets = composite_df['Dataset'].values
    models = ['Decision Trees', 'Gamification', 'ML', 'Proposed Model']
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, (model, color) in enumerate(zip(models, colors)):
        values = composite_df[model].values
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, values, width, label=model, color=color)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Composite Score')
    ax.set_title('Composite Equity-Accuracy-Engagement Score')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc='lower right')
    ax.set_ylim(0.5, 1.0)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_all_figures(result_dir: Path, output_dir: Path, table_data: Dict[str, pd.DataFrame]):
    """
    Generate all plots figures in folder.

    Args:
        result_dir: Directory containing result CSVs (or using table_data)
        output_dir: Directory to save figures
        table_data: Dictionary mapping table name to DataFrame with the data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use provided data or load from CSV if available
       
    if 'accuracy' in table_data:
        plot_accuracy_comparison(table_data['accuracy'], output_dir / 'accuracy.png')
    if 'disparity' in table_data:
        plot_disparity_reduction(table_data['disparity'], output_dir / 'disparity.png')
    if 'engagement' in table_data:
        plot_engagement_retention(table_data['engagement'], output_dir / 'engagement.png')
    if 'equity' in table_data:
        plot_equity_stability(table_data['equity'], output_dir / 'equity_stability.png')
    if 'divergence' in table_data:
        plot_counterfactual_divergence(table_data['divergence'], output_dir / 'counterfactual_divergence.png')
    if 'attention' in table_data:
        plot_attention_alignment(table_data['attention'], output_dir / 'attention_alignment.png')
    if 'pareto' in table_data:
        plot_pareto_balance(table_data['pareto'], output_dir / 'pareto_balance.png')
    if 'retention' in table_data:
        plot_policy_retention(table_data['retention'], output_dir / 'policy_retention.png')
    if 'composite' in table_data:
        plot_composite_scores(table_data['composite'], output_dir / 'composite.png')
    
    print(f"All figures saved to {output_dir}")
