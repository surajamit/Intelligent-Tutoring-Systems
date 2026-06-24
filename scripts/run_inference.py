"""
Inference script for deployment.

Author: Amit Pimpalkar
Organization: RBU, Nagpur
Year: 2026

Loads distilled policy and performs real-time prediction.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference with distilled policy')
    parser.add_argument('--model', type=str, required=True, help='Path to student policy checkpoint')
    parser.add_argument('--input', type=str, required=True, help='JSON file with input trajectory')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml')
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load input
    with open(args.input, 'r') as f:
        input_data = json.load(f)
    
    # Convert to tensors (simplified)
    states = torch.tensor(input_data['states'], dtype=torch.float32).unsqueeze(0).to(device)
    demographics = torch.tensor(input_data['demographics'], dtype=torch.long).unsqueeze(0).to(device)
    engagement = torch.tensor(input_data['engagement'], dtype=torch.float32).unsqueeze(0).to(device)
    
    logger.info(f"Input trajectory: {states.shape} states")
    
    # Load model
    import yaml
    config = yaml.safe_load(open(args.config))
    
    from src.models import IntegratedMultiObjectiveTutor
    model = IntegratedMultiObjectiveTutor(config).to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Run inference
    with torch.no_grad():
        outputs = model(
            states=states,
            demographic_ids=demographics,
            engagement=engagement,
            mask=torch.ones_like(states[:, :, 0])
        )
    
    predictions = outputs['predictions'].squeeze().cpu().numpy()
    
    logger.info(f"Prediction: {predictions.tolist()}")
    
    # Save results
    output = {
        'predictions': predictions.tolist(),
        'dis_signal': outputs['dis_signal'].item(),
        'divergence': outputs['divergence'].item()
    }
    
    with open('inference_output.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info("Inference complete. Results saved to inference_output.json")


if __name__ == '__main__':
    main()