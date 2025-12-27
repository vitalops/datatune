def _is_ibis_table(obj):
    try:
        import ibis
        return isinstance(obj, ibis.Table)
    except ImportError:
        return False
    
def _is_dask_df(obj):
    try:
        import dask.dataframe as dd
        return isinstance(obj, dd.DataFrame)
    except ImportError:
        return False

def filter(*, prompt, input_fields=None):
    def apply(llm, data):

        if _is_dask_df(data):
            from .dask.filter_dask import _filter_dask
            return _filter_dask(
                prompt=prompt,
                input_fields=input_fields,
            )(llm, data)
        elif _is_ibis_table(data):
            from .ibis.filter_ibis import _filter_ibis
            return _filter_ibis(
                prompt=prompt,
                input_fields=input_fields,
            )(llm, data)
        raise TypeError(f"Unsupported data type: {type(data)}")

    return apply


