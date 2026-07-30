class PA3EOModule(nn.Module):
    """
    Pareto-Adaptive Equity-Accuracy-Engagement Optimizer.
    Dynamically balances three objectives on the probability simplex.
    """
    
    def __init__(self, meta_lr=1e-3, update_frequency=5):
        super().__init__()
        self.meta_lr = meta_lr
        self.update_frequency = update_frequency
        self.epoch_counter = 0
        
        # Initial objective weights (sum to 1)
        self.lambda_acc = nn.Parameter(torch.tensor(0.4))
        self.lambda_eq = nn.Parameter(torch.tensor(0.3))
        self.lambda_eng = nn.Parameter(torch.tensor(0.3))
        
    def project_to_simplex(self, weights):
        """Project weights onto the probability simplex (sum = 1)."""
        # Sort in descending order
        u, _ = torch.sort(weights, descending=True)
        cssv = torch.cumsum(u, dim=0) - 1
        ind = torch.arange(u.size(0), device=u.device)
        cond = u - cssv / (ind + 1) > 0
        rho = ind[cond][-1] if cond.any() else 0
        theta = cssv[rho] / (rho + 1)
        return torch.clamp(weights - theta, min=0)
    
    def meta_gradient_update(self, losses, epoch):
        """
        Update objective weights using meta-gradient descent.
        
        Args:
            losses: Dictionary with 'acc', 'eq', 'eng' loss values
            epoch: Current epoch number
        Returns:
            lambda_new: Updated weights on simplex
        """
        self.epoch_counter += 1
        
        if self.epoch_counter % self.update_frequency != 0:
            # Return current weights without update
            return torch.stack([self.lambda_acc, self.lambda_eq, self.lambda_eng])
        
        # Current weights
        lambda_curr = torch.stack([self.lambda_acc, self.lambda_eq, self.lambda_eng])
        
        # Compute meta-gradient
        L_acc = losses['acc']
        L_eq = losses['eq']
        L_eng = losses['eng']
        
        L_total = lambda_curr[0] * L_acc + lambda_curr[1] * L_eq + lambda_curr[2] * L_eng
        
        # Compute gradient with respect to weights
        grad_lambda = torch.autograd.grad(L_total, lambda_curr, create_graph=True)[0]
        
        # Meta-gradient descent step
        lambda_new = lambda_curr - self.meta_lr * grad_lambda
        
        # Project onto simplex
        lambda_new = self.project_to_simplex(lambda_new)
        
        # Update parameters
        with torch.no_grad():
            self.lambda_acc.data = lambda_new[0]
            self.lambda_eq.data = lambda_new[1]
            self.lambda_eng.data = lambda_new[2]
        
        return lambda_new
    
    def forward(self, losses, epoch):
        """
        Compute total loss with dynamic weighting.
        
        Args:
            losses: Dictionary with 'acc', 'eq', 'eng' loss values
            epoch: Current epoch number
        Returns:
            L_total: Weighted total loss
            lambda_curr: Current objective weights
        """
        lambda_curr = self.meta_gradient_update(losses, epoch)
        
        L_total = (
            lambda_curr[0] * losses['acc'] +
            lambda_curr[1] * losses['eq'] +
            lambda_curr[2] * losses['eng']
        )
        
        return L_total, lambda_curr