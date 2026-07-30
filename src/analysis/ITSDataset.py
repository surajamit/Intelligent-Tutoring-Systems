import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

class ITSDataset(Dataset):
    """Dataset class for Intelligent Tutoring System with trajectory truncation."""
    
    def __init__(self, interactions_df, sequence_length=50, demographic_attributes=None):
        """
        Args:
            interactions_df: DataFrame with columns ['user_id', 'content_id', 'correct', 'timestamp', ...]
            sequence_length: T = 50 (truncated trajectory length)
            demographic_attributes: List of protected attribute columns (OULAD only)
        """
        self.sequence_length = sequence_length
        self.demographic_attributes = demographic_attributes
        self.users = interactions_df['user_id'].unique()
        self.trajectories = self._build_trajectories(interactions_df)
        
    def _build_trajectories(self, df):
        """Build user trajectories with truncation."""
        trajectories = []
        for user in self.users:
            user_data = df[df['user_id'] == user].sort_values('timestamp')
            if len(user_data) >= 50:
                # Truncate to last 50 interactions
                traj = user_data.iloc[-50:][['content_id', 'correct']].values
                trajectories.append(traj)
        return np.array(trajectories, dtype=np.float32)

def apply_gmm_clustering(embeddings, n_components=8):
    """
    Apply 8-component Gaussian Mixture Model to learner state embeddings.
    Used for EdNet and ASSISTments (no explicit demographics).
    """
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    clusters = gmm.fit_predict(embeddings_scaled)
    return clusters, gmm