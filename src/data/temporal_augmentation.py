"""
Temporal sequence augmentation for minority demographic groups.

Author: Amit Pimpalkar
Organization: RBU, Nagpur
Year: 2026

Implements cluster-consistent synthetic trajectory interpolation to correct
representational imbalances without compromising pedagogical coherence.
"""

import logging
from typing import List, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class TemporalSequenceAugmenter:
    """
    Applies temporal-aware synthetic oversampling for underrepresented cohorts.
    
    Unlike standard SMOTE which operates on independent samples, this approach:
    1. Identifies cluster-consistent neighbors in latent trajectory space
    2. Interpolates along the temporal dimension
    3. Injects controlled noise to prevent exact duplication
    """
    
    def __init__(
        self,
        k_neighbors: int = 5,
        augmentation_ratio: float = 0.15,
        noise_scale: float = 0.05
    ):
        """
        Initialize augmentation parameters.
        
        Args:
            k_neighbors: Number of nearest neighbors for interpolation
            augmentation_ratio: Target augmentation rate for minority class
            noise_scale: Gaussian noise standard deviation for synthetic samples
        """
        self.k_neighbors = k_neighbors
        self.augmentation_ratio = augmentation_ratio
        self.noise_scale = noise_scale
    
    def apply_augmentation(
        self,
        sequences: List[np.ndarray],
        demographics: List[np.ndarray],
        minority_indices: List[int]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Augment minority demographic trajectories.
        
        Args:
            sequences: List of temporal feature arrays [seq_len, feature_dim]
            demographics: Corresponding demographic vectors
            minority_indices: Indices of underrepresented samples
            
        Returns:
            Tuple of (augmented_sequences, augmented_demographics)
        """
        if not minority_indices:
            logger.info("No minority samples identified, skipping augmentation")
            return sequences, demographics
        
        logger.info(
            f"Augmenting {len(minority_indices)} minority samples "
            f"at ratio {self.augmentation_ratio}"
        )
        
        minority_sequences = [sequences[i] for i in minority_indices]
        minority_demographics = [demographics[i] for i in minority_indices]
        
        flattened = np.array([seq.flatten() for seq in minority_sequences])
        
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors, metric='euclidean')
        nbrs.fit(flattened)
        
        augmented_sequences = []
        augmented_demographics = []
        
        num_synthetic = int(len(minority_sequences) * self.augmentation_ratio)
        
        for _ in range(num_synthetic):
            seed_idx = np.random.randint(len(minority_sequences))
            seed_sequence = minority_sequences[seed_idx]
            seed_demo = minority_demographics[seed_idx]
            
            distances, indices = nbrs.kneighbors([flattened[seed_idx]])
            neighbor_idx = np.random.choice(indices[0][1:])
            
            neighbor_sequence = minority_sequences[neighbor_idx]
            
            interpolation_weight = np.random.uniform(0.3, 0.7)
            synthetic_sequence = (
                interpolation_weight * seed_sequence +
                (1 - interpolation_weight) * neighbor_sequence
            )
            
            noise = np.random.normal(0, self.noise_scale, synthetic_sequence.shape)
            synthetic_sequence = synthetic_sequence + noise
            
            synthetic_demo = seed_demo.copy()
            
            augmented_sequences.append(synthetic_sequence)
            augmented_demographics.append(synthetic_demo)
        
        combined_sequences = sequences + augmented_sequences
        combined_demographics = demographics + augmented_demographics
        
        logger.info(f"Created {len(augmented_sequences)} synthetic trajectories")
        
        return combined_sequences, combined_demographics