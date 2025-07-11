"""Gram matrix decoder package for analyzing neural network output neuron ordering."""

from .experiment import run_experiment
from .analysis import calculate_permutation_accuracy, plot_distance_distribution
from . import config

__all__ = ['run_experiment', 'calculate_permutation_accuracy', 'plot_distance_distribution', 'config']