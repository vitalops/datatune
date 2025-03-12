"""
Datatune: Your backend for LLM powered Big Data Apps
"""

__version__ = "0.1.0"

from .dataset import dataset, Dataset
from .operations import concat
from . import apps

__all__ = [
    "dataset",
    "Dataset",
    "apps",
    "concat",
]