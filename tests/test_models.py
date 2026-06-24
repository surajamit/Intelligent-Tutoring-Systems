"""
Unit tests for the core model components.

Author: Amit Pimpalkar
Organization: RBU, Nagpur
Year: 2026
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import torch
import yaml
import numpy as np

from src.models import (
    DemographicSensitivityGradientEncoder,
    CounterfactualEquityReplayNetwork,
    EngagementWeightedFairnessAttention,
    ParetoAdaptiveObjectiveController,
    IntegratedMultiObjectiveTutor
)


class TestDSGE(unittest.TestCase):
    def setUp(self):
        self.dsge = DemographicSensitivityGradientEncoder(smoothing_factor=0.05)
        self.batch_size = 4
        self.demo_dim = 8
        
    def test_forward_shape(self):
        loss = torch.tensor(1.0, requires_grad=True)
        demos = torch.randn(self.batch_size, self.demo_dim, requires_grad=True)
        dis, grad = self.dsge(loss, demos)
        self.assertEqual(dis.shape, torch.Size([]))
        self.assertEqual(grad.shape, torch.Size([self.batch_size, self.demo_dim]))


class TestCERN(unittest.TestCase):
    def setUp(self):
        self.cern = CounterfactualEquityReplayNetwork(num_counterfactual_samples=3)
        
    def test_divergence_computation(self):
        # Mock base model
        class MockModel(torch.nn.Module):
            def forward(self, x, d):
                return torch.randn(x.size(0), 10, 1)
        
        base = MockModel()
        states = torch.randn(4, 10, 5)
        factual_demos = torch.randn(4, 8)
        pool = torch.randn(20, 8)
        
        cf_out, div = self.cern(base, states, factual_demos, pool)
        self.assertEqual(cf_out.shape, torch.Size([4, 10, 1]))
        self.assertGreaterEqual(div.item(), 0.0)


class TestEWFAF(unittest.TestCase):
    def setUp(self):
        self.ewfaf = EngagementWeightedFairnessAttention(hidden_dim=16, num_attention_heads=2)
        
    def test_fusion_shape(self):
        hidden = torch.randn(2, 5, 16)
        engagement = torch.randn(2, 5)
        residuals = torch.randn(2, 5)
        output = self.ewfaf(hidden, engagement, residuals)
        self.assertEqual(output.shape, torch.Size([2, 5, 16]))


class TestParetoController(unittest.TestCase):
    def test_simplex_projection(self):
        controller = ParetoAdaptiveObjectiveController()
        weights = torch.tensor([1.0, 0.5, 0.2])
        proj = controller._project_simplex(weights)
        self.assertAlmostEqual(proj.sum().item(), 1.0, places=5)
        self.assertGreaterEqual(proj.min().item(), 0.0)
    
    def test_weight_update(self):
        controller = ParetoAdaptiveObjectiveController()
        controller.update_weights(0.1, 0.5, 0.3, epoch=5)
        self.assertAlmostEqual(controller.weights.sum().item(), 1.0, places=5)


class TestIntegratedModel(unittest.TestCase):
    def setUp(self):
        config = yaml.safe_load(open('configs/training_config.yaml'))
        self.model = IntegratedMultiObjectiveTutor(config)
        
    def test_forward_pass(self):
        batch_size = 2
        seq_len = 10
        state_dim = 5
        
        states = torch.randn(batch_size, seq_len, state_dim)
        demos = torch.randint(0, 10, (batch_size,))
        engagement = torch.randn(batch_size, seq_len)
        mask = torch.ones(batch_size, seq_len)
        
        outputs = self.model(states, demos, engagement, mask)
        self.assertIn('predictions', outputs)
        self.assertEqual(outputs['predictions'].shape, torch.Size([batch_size, seq_len, 1]))


if __name__ == '__main__':
    unittest.main()