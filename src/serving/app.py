#!/usr/bin/env python
"""
FastAPI deployment server for the distilled student policy.

Author: Amit Pimpalkar
Organization: RBU, Nagpur
Year: 2026

Serves real-time predictions for intelligent tutoring systems.
"""

import logging
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import yaml

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import IntegratedMultiObjectiveTutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Equity-Aware Tutoring API",
              description="Serves distilled policy for real-time mastery prediction",
              version="1.0.0")

# Global model
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TrajectoryRequest(BaseModel):
    """Input format for a single learner trajectory."""
    states: List[List[float]]  # [seq_len, feature_dim]
    demographics: List[int]    # [1] demographic group id
    engagement: List[float]    # [seq_len]
    mask: Optional[List[int]]  # [seq_len] binary mask


class PredictionResponse(BaseModel):
    """Prediction output."""
    predictions: List[float]
    dis_signal: float
    divergence: float
    status: str


@app.on_event("startup")
async def load_model():
    """Load the distilled policy on startup."""
    global model
    model_path = Path("outputs/distilled/student_policy.pth")
    config_path = Path("configs/training_config.yaml")
    
    if not model_path.exists():
        logger.warning(f"Model not found at {model_path}. Using dummy model.")
        # Create dummy for testing
        config = yaml.safe_load(open(config_path))
        model = IntegratedMultiObjectiveTutor(config).to(device)
    else:
        config = yaml.safe_load(open(config_path))
        model = IntegratedMultiObjectiveTutor(config).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logger.info(f"Model loaded from {model_path}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": str(device)}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TrajectoryRequest):
    """Run inference on a single trajectory."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to tensors
        states = torch.tensor(request.states, dtype=torch.float32).unsqueeze(0).to(device)
        demographics = torch.tensor(request.demographics, dtype=torch.long).unsqueeze(0).to(device)
        engagement = torch.tensor(request.engagement, dtype=torch.float32).unsqueeze(0).to(device)
        
        if request.mask:
            mask = torch.tensor(request.mask, dtype=torch.float32).unsqueeze(0).to(device)
        else:
            mask = torch.ones_like(states[:, :, 0])
        
        with torch.no_grad():
            outputs = model(states, demographics, engagement, mask)
        
        predictions = outputs['predictions'].squeeze().cpu().numpy().tolist()
        
        return PredictionResponse(
            predictions=predictions,
            dis_signal=outputs['dis_signal'].item(),
            divergence=outputs['divergence'].item(),
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/metrics")
async def get_model_metrics():
    """Return compressed model size and equity retention info."""
    if model is None:
        return {"error": "Model not loaded"}
    
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "parameters": total_params,
        "device": str(device),
        "status": "loaded"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)