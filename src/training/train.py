# train.py
import torch
from torch.utils.data import DataLoader
import wandb  # Optional: for experiment tracking

def main():
    # Configuration
    config = {
        'input_dim': 128,
        'hidden_dim': 256,
        'embedding_dim': 128,
        'demographic_dim': 8,
        'num_heads': 4,
        'head_dim': 32,
        'dropout': 0.2,
        'batch_size': 64,
        'learning_rate': 3e-4,
        'weight_decay': 1e-2,
        'epochs': 15,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Initialize framework
    framework = FairnessAwareITSFramework(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        embedding_dim=config['embedding_dim'],
        demographic_dim=config['demographic_dim'],
        num_heads=config['num_heads'],
        head_dim=config['head_dim'],
        dropout=config['dropout'],
        device=config['device']
    )
    
    # Load dataset (example: EdNet)
    # sequences, user_cluster_map = prepare_ednet('./data/ednet')
    
    # Create DataLoader
    # dataset = ITSDataset(sequences, demographic_ids=user_cluster_map)
    # dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Training loop
    for epoch in range(config['epochs']):
        epoch_losses = framework.train_epoch(dataloader, epoch)
        
        # Log metrics
        avg_losses = {
            k: np.mean([l[k] for l in epoch_losses]) 
            for k in epoch_losses[0].keys()
        }
        
        print(f"Epoch {epoch+1}/{config['epochs']}: "
              f"Loss={avg_losses['total']:.4f}, "
              f"Acc={avg_losses['accuracy']:.4f}, "
              f"Eq={avg_losses['equity']:.4f}, "
              f"Eng={avg_losses['engagement']:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': framework.encoder.state_dict(),
                'dsge_state_dict': framework.dsge.state_dict(),
                'cern_state_dict': framework.cern.state_dict(),
                'ewfaf_state_dict': framework.ewfaf.state_dict(),
                'pa3eo_state_dict': framework.pa3eo.state_dict(),
                'optimizer_state_dict': framework.optimizer.state_dict(),
                'losses': avg_losses
            }, f'checkpoint_epoch_{epoch+1}.pt')
    
    # Save final model
    torch.save({
        'encoder_state_dict': framework.encoder.state_dict(),
        'dsge_state_dict': framework.dsge.state_dict(),
        'cern_state_dict': framework.cern.state_dict(),
        'ewfaf_state_dict': framework.ewfaf.state_dict(),
        'pa3eo_state_dict': framework.pa3eo.state_dict(),
        'config': config
    }, 'final_model.pt')
    
    print("Training complete! Model saved as 'final_model.pt'")

if __name__ == '__main__':
    main()