"""
Datatune: Your backend for LLM powered Big Data Apps
==================================================

Datatune provides a unified interface for working with large datasets
and building LLM-powered data applications.
"""


from .dataset import dataset, Dataset
from . import apps

__all__ = [
    "dataset",
    "Dataset",
    "apps",
]