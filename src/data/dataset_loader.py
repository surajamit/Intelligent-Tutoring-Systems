"""
Educational dataset loading and trajectory construction.

Author: Amit Pimpalkar
Organization: RBU, Nagpur
Year: 2026

This module constructs temporally ordered interaction sequences from raw educational
logs, associating each trajectory with demographic context and engagement signals.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TutoringDataset(Dataset):
    """
    Manages sequential learner interaction data for deep learning consumption.
    
    Each sample represents a complete learning trajectory consisting of:
    - Temporal sequence of interaction states
    - Associated demographic embedding vector
    - Engagement proxy signals (persistence, latency, voluntary practice)
    - Binary mastery outcomes
    
    Attributes:
        trajectories: List of preprocessed interaction sequences
        demographics: Demographic context tensors
        engagement_signals: Extracted engagement metrics
        targets: Binary learning outcome labels
        max_length: Fixed sequence length for batch padding
    """
    
    def __init__(
        self,
        data_frame: pd.DataFrame,
        demographic_cols: List[str],
        feature_cols: List[str],
        max_length: int = 50,
        demographic_encoder: Optional[Dict] = None
    ):
        """
        Initialize dataset from preprocessed dataframe.
        
        Args:
            data_frame: Pandas dataframe with learner_id, timestamp, features
            demographic_cols: Column names for demographic attributes
            feature_cols: Column names for interaction features
            max_length: Maximum sequence length for truncation/padding
            demographic_encoder: Pre-fitted label encoders for categorical demographics
        """
        self.max_length = max_length
        self.feature_cols = feature_cols
        self.demographic_cols = demographic_cols
        
        self.trajectories, self.demographics, self.engagement_signals, self.targets = (
            self._construct_trajectories(data_frame, demographic_encoder)
        )
        
        logger.info(
            f"Constructed {len(self.trajectories)} trajectories. "
            f"Feature dim: {len(feature_cols)}, Demo dim: {len(demographic_cols)}"
        )
    
    def _construct_trajectories(
        self,
        df: pd.DataFrame,
        demographic_encoder: Optional[Dict]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Transform flat interaction log into grouped trajectories.
        
        For each learner:
        1. Sort interactions chronologically
        2. Extract feature sequences
        3. Compute engagement proxies from temporal patterns
        4. Associate demographic context
        
        Args:
            df: Input dataframe with learner_id and timestamp columns
            demographic_encoder: Mapping for categorical demographic encoding
            
        Returns:
            Tuple of (features, demographics, engagement, targets) lists
        """
        trajectories = []
        demographics = []
        engagement_signals = []
        targets = []
        
        for learner_id, group in df.groupby('learner_id'):
            group = group.sort_values('timestamp')
            
            feature_sequence = group[self.feature_cols].values
            
            if len(feature_sequence) > self.max_length:
                feature_sequence = feature_sequence[:self.max_length]
            else:
                padding_length = self.max_length - len(feature_sequence)
                feature_sequence = np.vstack([
                    feature_sequence,
                    np.zeros((padding_length, len(self.feature_cols)))
                ])
            
            demographic_vector = group.iloc[0][self.demographic_cols].values
            
            engagement = self._compute_engagement_proxies(group)
            
            target_sequence = group['correctness'].values
            if len(target_sequence) > self.max_length:
                target_sequence = target_sequence[:self.max_length]
            else:
                target_sequence = np.pad(
                    target_sequence,
                    (0, self.max_length - len(target_sequence)),
                    constant_values=0
                )
            
            trajectories.append(feature_sequence.astype(np.float32))
            demographics.append(demographic_vector.astype(np.float32))
            engagement_signals.append(engagement.astype(np.float32))
            targets.append(target_sequence.astype(np.float32))
        
        return trajectories, demographics, engagement_signals, targets
    
    def _compute_engagement_proxies(self, interaction_group: pd.DataFrame) -> np.ndarray:
        """
        Derive continuous engagement score from behavioral patterns.
        
        Combines:
        - Session persistence (length of uninterrupted activity)
        - Response latency (inverse of mean time-on-task)
        - Voluntary practice frequency (sessions beyond required minimum)
        
        Args:
            interaction_group: Single learner's temporal interaction sequence
            
        Returns:
            Normalized engagement score vector of length max_length
        """
        session_lengths = interaction_group.groupby('session_id').size().values
        persistence = np.mean(session_lengths) if len(session_lengths) > 0 else 0.0
        
        if 'response_time' in interaction_group.columns:
            mean_latency = interaction_group['response_time'].mean()
            latency_score = 1.0 / (1.0 + mean_latency / 1000.0)
        else:
            latency_score = 0.5
        
        required_min = 10
        actual_interactions = len(interaction_group)
        voluntary_score = min(1.0, (actual_interactions - required_min) / required_min) if actual_interactions > required_min else 0.0
        
        engagement_score = 0.4 * persistence + 0.3 * latency_score + 0.3 * voluntary_score
        
        engagement_vector = np.full(self.max_length, engagement_score)
        
        return engagement_vector
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve single trajectory sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - states: [seq_len, feature_dim] interaction features
                - demographics: [demo_dim] demographic embedding
                - engagement: [seq_len] engagement proxies
                - targets: [seq_len] binary correctness labels
                - mask: [seq_len] valid timestep indicators (for padding)
        """
        trajectory = self.trajectories[idx]
        demographic = self.demographics[idx]
        engagement = self.engagement_signals[idx]
        target = self.targets[idx]
        
        mask = (trajectory.sum(axis=1) != 0).astype(np.float32)
        
        return {
            'states': torch.from_numpy(trajectory),
            'demographics': torch.from_numpy(demographic),
            'engagement': torch.from_numpy(engagement),
            'targets': torch.from_numpy(target),
            'mask': torch.from_numpy(mask)
        }


def load_educational_dataset(
    dataset_name: str,
    config: Dict,
    split: str = 'train'
) -> TutoringDataset:
    """
    Load and preprocess educational interaction dataset.
    
    Args:
        dataset_name: One of ['ednet', 'assistments', 'oulad']
        config: Dataset configuration dictionary
        split: Data split ['train', 'val', 'test']
        
    Returns:
        Initialized TutoringDataset instance
        
    Raises:
        ValueError: If dataset_name is not recognized
        FileNotFoundError: If required data files are missing
    """
    dataset_config = config.get(dataset_name)
    if dataset_config is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    processed_dir = Path(dataset_config['processed_dir'])
    split_file = processed_dir / f"{split}_sequences.csv"
    
    if not split_file.exists():
        raise FileNotFoundError(
            f"Processed data not found: {split_file}. "
            f"Run data preparation script first."
        )
    
    logger.info(f"Loading {dataset_name} {split} split from {split_file}")
    df = pd.read_csv(split_file)
    
    dataset = TutoringDataset(
        data_frame=df,
        demographic_cols=dataset_config['demographic_columns'],
        feature_cols=dataset_config['feature_columns'],
        max_length=dataset_config['max_sequence_length']
    )
    
    return dataset