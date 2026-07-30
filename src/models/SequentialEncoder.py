import torch.nn as nn
import torch.nn.functional as F

class SequentialEncoder(nn.Module):
    """
    GRU-based sequential encoder with demographic conditioning.
    Hidden dimension h=256, embedding dimension d=128.
    """
    
    def __init__(self, input_dim, hidden_dim=256, embedding_dim=128, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        self.input_projection = nn.Linear(input_dim, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, demographic_embedding=None):
        """
        Args:
            x: Input sequence of shape (batch, T, input_dim)
            demographic_embedding: Optional demographic embedding for conditioning
        Returns:
            hidden_states: (batch, T, hidden_dim)
        """
        x = self.input_projection(x)
        x = self.dropout(x)
        
        # Condition on demographic if provided (DSGE stage)
        if demographic_embedding is not None:
            # Expand demographic embedding to match sequence length
            demo_expanded = demographic_embedding.unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, demo_expanded], dim=-1)
            # Project back to embedding dimension
            x = self.input_projection(x)
        
        hidden_states, _ = self.gru(x)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states