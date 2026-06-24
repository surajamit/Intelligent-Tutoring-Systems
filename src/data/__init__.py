"""
Data processing module for equity-aware tutoring system.

Author: Amit Pimpalkar
Organization: RBU, Nagpur
Year: 2026
"""

from .dataset_loader import TutoringDataset, load_educational_dataset
from .preprocessor import DataPreprocessor
from .temporal_augmentation import TemporalSequenceAugmenter
from .batch_sampler import EquityAwareBatchSampler

__all__ = [
    'TutoringDataset',
    'load_educational_dataset',
    'DataPreprocessor',
    'TemporalSequenceAugmenter',
    'EquityAwareBatchSampler'
]