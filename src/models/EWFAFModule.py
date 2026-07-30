class EWFAFModule(nn.Module):
    """
    Engage-Weighted Fairness Attention Fusion.
    Fairness-gated attention mechanism balancing engagement and equity.
    """
    
    def __init__(self, hidden_dim, num_heads=4, head_dim=32, fairness_regularization=0.01):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rho = fairness_regularization
        
        # Multi-head attention components
        self.W_q = nn.Linear(hidden_dim, num_heads * head_dim)
        self.W_k = nn.Linear(hidden_dim, num_heads * head_dim)
        self.W_v = nn.Linear(hidden_dim, num_heads * head_dim)
        self.W_o = nn.Linear(num_heads * head_dim, hidden_dim)
        
        # Engagement-equity scaling parameters
        self.W_e = nn.Parameter(torch.ones(1))
        self.W_f = nn.Parameter(torch.ones(1))
        
    def compute_attention_weights(self, hidden_states, engagement_scores, fairness_risk):
        """
        Compute fairness-gated attention weights.
        
        Args:
            hidden_states: (batch, T, hidden_dim)
            engagement_scores: (batch, T) - normalized engagement scores [0,1]
            fairness_risk: (batch, T) - counterfactual risk at each step
        Returns:
            alpha: Attention weights (batch, T)
            ℒ_eng_eq: Engagement-equity regularization loss
        """
        # Compute base attention scores
        Q = self.W_q(hidden_states).view(hidden_states.size(0), -1, self.num_heads, self.head_dim)
        K = self.W_k(hidden_states).view(hidden_states.size(0), -1, self.num_heads, self.head_dim)
        V = self.W_v(hidden_states).view(hidden_states.size(0), -1, self.num_heads, self.head_dim)
        
        # Scaled dot-product attention
        scores = torch.einsum('bthd,bthd->bth', Q, K) / np.sqrt(self.head_dim)
        
        # Apply fairness gating: scale down attention for high-risk steps
        gating = torch.sigmoid(self.W_e * engagement_scores - self.W_f * fairness_risk)
        scores = scores * gating
        
        # Softmax over time dimension
        alpha = F.softmax(scores, dim=1)
        
        # Engagement-equity regularization
        # Penalizes extreme variation in attention with respect to fairness risk
        dalpha_drisk = torch.autograd.grad(
            alpha.sum(), fairness_risk, create_graph=True, retain_graph=True
        )[0]
        ℒ_eng_eq = torch.mean(dalpha_drisk ** 2)
        
        return alpha, ℒ_eng_eq
    
    def forward(self, hidden_states, engagement_scores, fairness_risk):
        """
        Forward pass with fairness-gated attention.
        
        Returns:
            z: Fused representation vector
            ℒ_eng_eq: Engagement-equity regularization loss
        """
        alpha, ℒ_eng_eq = self.compute_attention_weights(
            hidden_states, engagement_scores, fairness_risk
        )
        
        # Weighted fusion
        z = torch.einsum('bth,bthd->btd', alpha, hidden_states)
        z = self.W_o(z)
        
        return z, ℒ_eng_eq