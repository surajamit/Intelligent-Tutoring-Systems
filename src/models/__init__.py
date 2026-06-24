"""
Neural architecture components for equity-aware tutoring.

Author: Amit Pimpalkar
Organization: RBU, Nagpur
Year: 2026
"""

from .dsge_module import DemographicSensitivityGradientEncoder
from .cern_module import CounterfactualEquityReplayNetwork
from .ewfaf_module import EngagementWeightedFairnessAttention
from .pareto_controller import ParetoAdaptiveObjectiveController
from .policy_distillation import EquityPreservingDistillation
from .integrated_tutor import IntegratedMultiObjectiveTutor

__all__ = [
    'DemographicSensitivityGradientEncoder',
    'CounterfactualEquityReplayNetwork',
    'EngagementWeightedFairnessAttention',
    'ParetoAdaptiveObjectiveController',
    'EquityPreservingDistillation',
    'IntegratedMultiObjectiveTutor'
]