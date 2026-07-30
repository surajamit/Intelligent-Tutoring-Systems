# evaluate.py
import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def evaluate_model(model_path, test_dataloader, device='cuda'):
    """
    Load trained model and evaluate on test set.
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Rebuild framework
    framework = FairnessAwareITSFramework(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        embedding_dim=config['embedding_dim'],
        demographic_dim=config['demographic_dim'],
        num_heads=config['num_heads'],
        head_dim=config['head_dim'],
        dropout=config['dropout'],
        device=device
    )
    
    # Load weights
    framework.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    framework.dsge.load_state_dict(checkpoint['dsge_state_dict'])
    framework.cern.load_state_dict(checkpoint['cern_state_dict'])
    framework.ewfaf.load_state_dict(checkpoint['ewfaf_state_dict'])
    framework.pa3eo.load_state_dict(checkpoint['pa3eo_state_dict'])
    
    framework.eval()
    
    all_preds = []
    all_labels = []
    all_equity_losses = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            trajectory = batch['trajectory'].to(device)
            demographic_ids = batch['demographic_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            hidden_states = framework.encoder(trajectory)
            pred = framework.encoder.input_projection(hidden_states[:, -1, :])
            
            # Compute metrics
            all_preds.extend(torch.argmax(pred, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Compute equity loss
            S_hat = framework.dsge.compute_sensitivity(
                F.cross_entropy(pred, labels), 
                demographic_ids, 
                framework.encoder.parameters()
            )
            
            # Simplified equity evaluation
            unique_demos = torch.unique(demographic_ids)
            demo_accuracies = []
            for demo in unique_demos:
                mask = demographic_ids == demo
                if mask.sum() > 0:
                    demo_preds = torch.argmax(pred[mask], dim=1)
                    demo_labels = labels[mask]
                    demo_acc = (demo_preds == demo_labels).float().mean()
                    demo_accuracies.append(demo_acc.item())
            
            if len(demo_accuracies) > 1:
                equity_variance = np.var(demo_accuracies)
                all_equity_losses.append(equity_variance)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    avg_equity_variance = np.mean(all_equity_losses) if all_equity_losses else 0
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test AUC: {auc:.4f}")
    print(f"Avg Equity Variance: {avg_equity_variance:.4f}")
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'equity_variance': avg_equity_variance
    }

# Performance metrics computation (Tables 4-10)
def compute_all_metrics(framework, dataloader, device='cuda'):
    """
    Compute all metrics reported in Tables 4-10.
    """
    results = {
        'accuracy': [],
        'disparity': [],
        'esi': [],
        'ctda': [],
        'engagement': [],
        'po bs': [],
        'afas': []
    }
    
    # Implementation of each metric as per manuscript
    # ...
    
    return results