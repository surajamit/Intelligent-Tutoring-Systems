class DSGEModule(nn.Module):
    """
    Demographic Sensitivity Gradient Encoding.
    Computes gradient-level sensitivities and stabilizes with Hessian approximation.
    """
    
    def __init__(self, hidden_dim, demographic_dim, smoothing_weight=0.05):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.demographic_dim = demographic_dim
        self.lambda_1 = smoothing_weight
        
        # Demographic embedding layer
        self.demographic_embedding = nn.Embedding(demographic_dim, hidden_dim)
        
    def compute_sensitivity(self, loss, demographic_ids, model_params, m=5):
        """
        Compute stabilized demographic sensitivity Ŝ_d using diagonal Fisher approximation.
        
        Args:
            loss: Learning loss ℒ_learn
            demographic_ids: Demographic group IDs for current batch
            model_params: Model parameters requiring gradient
            m: Number of Hutchinson projections (default: 5)
        Returns:
            S_hat: Stabilized sensitivity scalar
        """
        # Compute gradient with respect to demographic embedding
        demo_emb = self.demographic_embedding(demographic_ids)
        grad_demo = torch.autograd.grad(
            loss, demo_emb, create_graph=True, retain_graph=True
        )[0]
        
        # Hutchinson trace estimator for Hessian diagonal
        hessian_trace = 0.0
        for _ in range(m):
            v = torch.randint(0, 2, grad_demo.shape, device=grad_demo.device).float() * 2 - 1
            grad_v = torch.sum(grad_demo * v)
            hessian_v = torch.autograd.grad(grad_v, demo_emb, retain_graph=True)[0]
            hessian_trace += torch.sum(hessian_v * v)
        
        hessian_trace = hessian_trace / m
        
        # Compute Frobenius norm approximation
        hessian_frobenius = torch.sqrt(torch.sum(hessian_trace ** 2))
        
        # Stabilized sensitivity
        S_hat = torch.norm(grad_demo) ** 2 + self.lambda_1 * hessian_frobenius
        
        return S_hat