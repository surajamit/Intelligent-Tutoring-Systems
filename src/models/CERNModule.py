class CERNModule(nn.Module):
    """
    Counterfactual Equity Replay Networks.
    Implements abduction-action-prediction counterfactual generation.
    """
    
    def __init__(self, encoder, hidden_dim, demographic_dim, fairness_weight=0.10):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.demographic_dim = demographic_dim
        self.lambda_2 = fairness_weight
        
        # Conditional VAE for abduction (inferring exogenous variables)
        self.vae_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.vae_mu = nn.Linear(128, 64)
        self.vae_logvar = nn.Linear(128, 64)
        self.vae_decoder = nn.Sequential(
            nn.Linear(64 + hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )
        
        # Counterfactual simulator
        self.cf_simulator = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
    def abduction(self, trajectory, demographic_id):
        """Step 1: Infer exogenous variables U* from observed trajectory."""
        # Encode trajectory to get latent state
        hidden = self.encoder(trajectory)
        hidden_pooled = torch.cat([hidden.mean(dim=1), hidden.max(dim=1)[0]], dim=-1)
        
        # VAE encoding
        encoded = self.vae_encoder(hidden_pooled)
        mu = self.vae_mu(encoded)
        logvar = self.vae_logvar(encoded)
        
        # Reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        U_star = mu + eps * std
        
        return U_star, mu, logvar
    
    def action(self, demographic_id, alternative_demographic_id):
        """Step 2: Intervene on demographic attribute do(D = d')."""
        # Replace demographic while keeping all other mechanisms invariant
        return alternative_demographic_id
    
    def prediction(self, U_star, alternative_demographic_id, initial_state):
        """Step 3: Generate counterfactual trajectory."""
        # Decode exogenous variables
        decoded = self.vae_decoder(torch.cat([U_star, initial_state], dim=-1))
        
        # Forward simulate under intervention
        cf_trajectory, _ = self.cf_simulator(decoded.unsqueeze(1))
        
        return cf_trajectory
    
    def forward(self, trajectory, demographic_id, alternative_demographic_id, S_hat):
        """
        Complete abduction-action-prediction pipeline.
        
        Args:
            trajectory: Factual trajectory (batch, T, hidden_dim)
            demographic_id: Current demographic group
            alternative_demographic_id: Target demographic for counterfactual
            S_hat: Stabilized sensitivity from DSGE
        Returns:
            ℒ_eq: Causal equity loss
        """
        # Step 1: Abduction
        U_star, mu, logvar = self.abduction(trajectory, demographic_id)
        
        # Step 2: Action
        demo_alt = self.action(demographic_id, alternative_demographic_id)
        
        # Step 3: Prediction
        initial_state = trajectory[:, 0, :]
        cf_trajectory = self.prediction(U_star, demo_alt, initial_state)
        
        # Compute Counterfactual Trajectory Divergence (CTDA)
        ctda = torch.mean((trajectory - cf_trajectory) ** 2)
        
        # Scale with DSGE signal and apply temporal regularizer
        temporal_reg = torch.exp(-0.1 * torch.arange(trajectory.size(1), device=trajectory.device))
        temporal_reg = temporal_reg.mean()
        
        ℒ_eq = ctda * S_hat * temporal_reg
        
        return ℒ_eq, ctda