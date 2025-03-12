from ..dataset import Dataset

class BaseApp:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset