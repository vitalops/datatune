import pytest
import ibis
import pandas as pd
import datatune as dt


def mock_llm(input_list, *args, **kwargs):
    return ['{"Type": "Fruit"}'] * len(input_list)

def mock_filter_llm(input_list, *args, **kwargs):
    responses = []
    for item in input_list:

        is_veg = "true" if "carrot" in item or "onion" in item else "false"
        responses.append(f'{{"__filter__": "{is_veg}"}}')
    return responses

@pytest.fixture(params=['duckdb', 'datafusion', 'sqlite'])
def con(request):
    if request.param == 'duckdb':
        return ibis.duckdb.connect()
    elif request.param == 'datafusion':
        return ibis.datafusion.connect()
    elif request.param == 'sqlite':
        return ibis.sqlite.connect()

def test_map_logic(con):
    df = pd.DataFrame({"item": ["apple", "banana","carrot","onion"], "price": [1, 2, 3, 4]})
    table = con.create_table("test_data", df)

    llm = mock_llm

    mapped = dt.map(
    prompt = "create column type based on if item is fruit or vegetable",
    output_fields=["Type"],
    input_fields=["item"]
     )(llm, table)
    
    result = mapped.execute()
    assert "Type" in result.columns
    assert "item" in result.columns
    assert "price" in result.columns
    assert all(result["Type"] == "Fruit")
    assert len(result) == 4

def test_filter_logic(con):
    df = pd.DataFrame({"item": ["apple", "banana","carrot","onion"], "price": [1, 2, 3, 4]})
    table = con.create_table("test_data", df)

    llm = mock_filter_llm

    filtered = dt.filter(
    prompt = "only keep vegetables",
    input_fields=["item"]
     )(llm, table)
    
    result = filtered.execute()
    surviving_items = result["item"].tolist()
    assert "carrot" in surviving_items
    assert "onion" in surviving_items
    assert "apple" not in surviving_items
    assert len(result) == 2
