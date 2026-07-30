class EPPDVModule(nn.Module):
    """
    Equity-Preserving Policy Distillation and Validation.
    Compresses teacher network while preserving equity guarantees.
    """
    
    def __init__(self, teacher, student, temperature=2.0, equity_penalty=0.5):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.gamma = equity_penalty
        
    def distill(self, batch, demographic_ids):
        """
        Perform knowledge distillation with equity retention.
        
        Args:
            batch: Input batch (x, actions, rewards, ...)
            demographic_ids: Demographic group IDs
        Returns:
            ℒ_distill: Distillation loss
            ERR: Equity Retention Ratio
        """
        # Teacher predictions
        with torch.no_grad():
            teacher_logits = self.teacher(batch)
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # Student predictions
        student_logits = self.student(batch)
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # Imitation loss (KL divergence)
        ℒ_imitation = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
        
        # Student equity loss (reuse CERN module)
        ℒ_eq_student = self.compute_equity_loss(student_logits, demographic_ids)
        
        # Total distillation loss
        ℒ_distill = ℒ_imitation + self.gamma * ℒ_eq_student
        
        return ℒ_distill, ℒ_eq_student
    
    def compute_equity_loss(self, logits, demographic_ids):
        """Compute equity loss for the student model."""
        # Simplified equity loss - measures variance across demographic groups
        unique_demos = torch.unique(demographic_ids)
        group_means = []
        for demo in unique_demos:
            mask = demographic_ids == demo
            group_logits = logits[mask]
            if len(group_logits) > 0:
                group_means.append(group_logits.mean(dim=0))
        
        if len(group_means) > 1:
            group_means = torch.stack(group_means)
            equity_loss = torch.var(group_means, dim=0).mean()
        else:
            equity_loss = torch.tensor(0.0, device=logits.device)
        
        return equity_loss
    
    def compute_equity_retention_ratio(self, teacher_eq_loss, student_eq_loss):
        """
        Compute Equity Retention Ratio (ERR).
        Must be ≥ 0.95 for successful distillation.
        """
        ERR = student_eq_loss / (teacher_eq_loss + 1e-8)
        return ERR