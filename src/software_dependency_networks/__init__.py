"""Simulation and reporting tools for software dependency network avalanches."""

from .model import DependencyNetworkSimulation, NetworkConfig
from .pipeline import ExperimentConfig, run_experiment
from .reporting import build_report_pdf

__all__ = [
    "DependencyNetworkSimulation",
    "ExperimentConfig",
    "NetworkConfig",
    "build_report_pdf",
    "run_experiment",
]
