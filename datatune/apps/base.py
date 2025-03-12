"""
Base app class for datatune.
"""

from ..dataset import Dataset

class BaseApp:
    """
    Base class for datatune apps.
    
    All apps should inherit from this class.
    """
    
    def __init__(self, dataset: Dataset):
        """
        Initialize a BaseApp.
        
        Args:
            dataset: Dataset to use with the app
        """
        self.dataset = dataset