class Dataset:
    def __init__(self, data=None):
        self.data = data if data is not None else {}   

def dataset(*args, **kwargs):
    return Dataset(*args, **kwargs)