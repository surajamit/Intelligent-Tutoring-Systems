"""
Stratified batch sampling to maintain demographic balance.

Author: Amit Pimpalkar
Organization: RBU, Nagpur
Year: 2026

Ensures each batch contains proportional representation across demographic groups.
"""

import logging
from typing import Iterator, List

import numpy as np
import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class EquityAwareBatchSampler(Sampler):
    """
    Custom batch sampler ensuring demographic stratification.
    
    Standard random sampling can produce batches dominated by majority groups,
    leading to gradient estimates biased toward majority patterns. This sampler
    enforces proportional demographic representation within each batch.
    """
    
    def __init__(
        self,
        demographic_labels: np.ndarray,
        batch_size: int,
        drop_last: bool = False
    ):
        """
        Initialize stratified sampler.
        
        Args:
            demographic_labels: Array of demographic group identifiers
            batch_size: Target batch size
            drop_last: Whether to drop final incomplete batch
        """
        self.demographic_labels = demographic_labels
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        self.group_indices = self._build_group_indices()
        
        self.num_batches = len(demographic_labels) // batch_size
        if not drop_last and len(demographic_labels) % batch_size != 0:
            self.num_batches += 1
    
    def _build_group_indices(self) -> dict:
        """Map demographic groups to sample indices."""
        groups = {}
        for idx, label in enumerate(self.demographic_labels):
            label = int(label)
            if label not in groups:
                groups[label] = []
            groups[label].append(idx)
        
        logger.info(f"Built {len(groups)} demographic strata")
        for group_id, indices in groups.items():
            logger.info(f"  Group {group_id}: {len(indices)} samples")
        
        return groups
    
    def __iter__(self) -> Iterator[List[int]]:
        """
        Iterate over stratified batches.
        
        Yields:
            List of sample indices forming a single batch
        """
        shuffled_groups = {
            group_id: np.random.permutation(indices).tolist()
            for group_id, indices in self.group_indices.items()
        }
        
        for batch_idx in range(self.num_batches):
            batch_indices = []
            
            samples_per_group = self.batch_size // len(shuffled_groups)
            remainder = self.batch_size % len(shuffled_groups)
            
            for group_idx, (group_id, indices) in enumerate(shuffled_groups.items()):
                n_samples = samples_per_group
                if group_idx < remainder:
                    n_samples += 1
                
                if not indices:
                    indices = self.group_indices[group_id].copy()
                    np.random.shuffle(indices)
                    shuffled_groups[group_id] = indices
                
                selected = indices[:n_samples]
                batch_indices.extend(selected)
                shuffled_groups[group_id] = indices[n_samples:]
            
            yield batch_indices
    
    def __len__(self) -> int:
        return self.num_batches