class FairnessAwareITSFramework:
    """
    Complete Multi-Objective Fairness-Aware Intelligent Tutoring System.
    """
    
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        embedding_dim=128,
        demographic_dim=8,
        num_heads=4,
        head_dim=32,
        dropout=0.2,
        device='cuda'
    ):
        self.device = device
        
        # Initialize all modules
        self.encoder = SequentialEncoder(input_dim, hidden_dim, embedding_dim, dropout)
        self.dsge = DSGEModule(hidden_dim, demographic_dim)
        self.cern = CERNModule(self.encoder, hidden_dim, demographic_dim)
        self.ewfaf = EWFAFModule(hidden_dim, num_heads, head_dim)
        self.pa3eo = PA3EOModule()
        
        # Move to device
        self.encoder.to(device)
        self.dsge.to(device)
        self.cern.to(device)
        self.ewfaf.to(device)
        self.pa3eo.to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) +
            list(self.dsge.parameters()) +
            list(self.cern.parameters()) +
            list(self.ewfaf.parameters()),
            lr=3e-4,
            weight_decay=1e-2
        )
        
        self.epoch = 0
        
    def train_step(self, batch, demographic_ids, engagement_scores):
        """
        Single training step with all five stages.
        
        Args:
            batch: (trajectory, actions, rewards, ...)
            demographic_ids: Demographic group IDs
            engagement_scores: Engagement scores per time step
        Returns:
            losses: Dictionary of all loss components
        """
        self.optimizer.zero_grad()
        
        # ---- STAGE 1: DSGE ----
        # Encode trajectory
        hidden_states = self.encoder(batch['trajectory'])
        
        # Compute learning loss (next-step prediction)
        pred = self.encoder.input_projection(hidden_states[:, -1, :])
        ℒ_learn = F.cross_entropy(pred, batch['labels'])
        
        # Compute demographic sensitivity
        S_hat = self.dsge.compute_sensitivity(ℒ_learn, demographic_ids, self.encoder.parameters())
        
        # ---- STAGE 2: CERN ----
        # Generate counterfactual trajectories
        alt_demo_ids = torch.randint(0, self.dsge.demographic_dim, demographic_ids.shape, device=self.device)
        ℒ_eq, ctda = self.cern(hidden_states, demographic_ids, alt_demo_ids, S_hat)
        
        # ---- STAGE 3: EWFAF ----
        # Fairness-gated attention
        fairness_risk = ctda.detach()  # Use CTDA as fairness risk proxy
        z, ℒ_eng_eq = self.ewfaf(hidden_states, engagement_scores, fairness_risk)
        
        # Engagement loss (negative: we want to maximize engagement)
        ℒ_eng = -torch.mean(engagement_scores)
        
        # ---- STAGE 4: PA³EO ----
        losses = {
            'acc': ℒ_learn,
            'eq': ℒ_eq,
            'eng': ℒ_eng
        }
        
        ℒ_total, lambda_curr = self.pa3eo(losses, self.epoch)
        
        # Add engagement-equity regularization
        ℒ_total = ℒ_total + self.ewfaf.rho * ℒ_eng_eq
        
        # ---- Backpropagation ----
        ℒ_total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) +
            list(self.dsge.parameters()) +
            list(self.cern.parameters()) +
            list(self.ewfaf.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
        
        self.epoch += 1
        
        return {
            'total': ℒ_total.item(),
            'accuracy': ℒ_learn.item(),
            'equity': ℒ_eq.item(),
            'engagement': ℒ_eng.item(),
            'eng_eq_reg': ℒ_eng_eq.item(),
            'ctda': ctda.item(),
            'lambda_acc': lambda_curr[0].item(),
            'lambda_eq': lambda_curr[1].item(),
            'lambda_eng': lambda_curr[2].item()
        }
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        self.encoder.train()
        self.dsge.train()
        self.cern.train()
        self.ewfaf.train()
        
        epoch_losses = []
        for batch in dataloader:
            # Move batch to device
            trajectory = batch['trajectory'].to(self.device)
            demographic_ids = batch['demographic_ids'].to(self.device)
            engagement_scores = batch['engagement_scores'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            batch_dict = {
                'trajectory': trajectory,
                'labels': labels
            }
            
            losses = self.train_step(batch_dict, demographic_ids, engagement_scores)
            epoch_losses.append(losses)
        
        return epoch_losses
    
    def distill(self, student_model, dataloader, target_err=0.95):
        """
        Perform equity-preserving distillation (EPPDV).
        
        Args:
            student_model: Lightweight student network
            dataloader: DataLoader for distillation
            target_err: Target Equity Retention Ratio (default: 0.95)
        Returns:
            student_model: Distilled student model
            err: Final Equity Retention Ratio
        """
        eppdv = EPPDVModule(self, student_model)
        
        student_optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)
        
        err = 0.0
        epoch = 0
        
        while err < target_err and epoch < 20:
            student_model.train()
            teacher_eq_losses = []
            student_eq_losses = []
            
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                demographic_ids = batch['demographic_ids']
                
                # Compute distillation loss
                ℒ_distill, ℒ_eq_student = eppdv.distill(batch['trajectory'], demographic_ids)
                
                # Backpropagate
                student_optimizer.zero_grad()
                ℒ_distill.backward()
                student_optimizer.step()
                
                # Compute teacher equity loss (for ERR)
                with torch.no_grad():
                    teacher_logits = self.encoder(batch['trajectory'])
                    ℒ_eq_teacher = eppdv.compute_equity_loss(teacher_logits, demographic_ids)
                    
                teacher_eq_losses.append(ℒ_eq_teacher.item())
                student_eq_losses.append(ℒ_eq_student.item())
            
            # Compute ERR
            err = np.mean(student_eq_losses) / (np.mean(teacher_eq_losses) + 1e-8)
            epoch += 1
            
            print(f"Distillation Epoch {epoch}: ERR = {err:.4f}")
        
        return student_model, err