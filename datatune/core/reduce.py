import dask.dataframe as dd
from .registry import get_action

_ACTIONS = {}

def register_action(name):
    def decorator(cls):
        _ACTIONS[name] = cls
        return cls
    return decorator

def get_action(name):
    try:
        return _ACTIONS[name]
    except KeyError:
        raise ValueError(f"Unknown action: {name}")

 
def reduce(df, action: str, **kwargs):
    cls = get_action(action)
    reducer = cls(**kwargs)   
    return reducer(df)   