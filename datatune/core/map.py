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

def map(*, prompt, output_fields, input_fields=None):
    def apply(llm, data):
        
        if _is_dask_df(data):
            from .dask.map_dask import _map_dask
            return _map_dask(
                prompt=prompt,
                output_fields=output_fields,
                input_fields=input_fields,
            )(llm, data)
        elif _is_ibis_table(data):
            from .ibis.map_ibis import _map_ibis
            return _map_ibis(
                prompt=prompt,
                output_fields=output_fields,
                input_fields=input_fields,
            )(llm, data)

        raise TypeError(f"Unsupported data type: {type(data)}")

    return apply


