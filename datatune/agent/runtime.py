import dask.dataframe as dd

class Runtime:
    def __init__(self):
        self.namespace = {}

    def execute(self, code: str):
        exec(code, self.namespace)

    def __getitem__(self, key: str):
        try:
            return self.namespace[key]
        except KeyError as e:
            raise NameError(f"Variable '{key}' is not defined in the runtime environment.") from e

    def __setitem__(self, key: str, value):
        self.namespace[key] = value

    def get(self, key: str, default=None):
        return self.namespace.get(key, default)
