from typing import Dict, List, Union

import dask.dataframe as dd
import pandas as pd

import datatune as dt
from datatune.core.constants import DELETED_COLUMN, ERRORED_COLUMN


class MockLLM:
    def __init__(self, responses: List[str]):
        self.responses = responses
        self.response_index = 0

    def __call__(self, prompt: Union[str, List[str]]) -> List[str]:
        """Standard call method for backward compatibility."""
        if isinstance(prompt, pd.Series):
            num_responses = len(prompt)
            return self.responses[:num_responses]
        elif isinstance(prompt, str):
            return [self.responses[0]]
        else:
            num_responses = len(prompt)
            return self.responses[:num_responses]


def create_test_dataframe():
    data = {
        "first_name": [
            "John",
            "Jane",
            "Bob",
            "Alice",
            "Tom",
            "Mary",
            "Steve",
            "Sarah",
            "Mike",
            "Lisa",
        ],
        "last_name": [
            "Smith",
            "Johnson",
            "Lee",
            "Williams",
            "Brown",
            "Davis",
            "Wilson",
            "Jones",
            "Taylor",
            "Miller",
        ],
        "email": [
            "john@example.com",
            "jane@example.com",
            "bob@example.com",
            "alice@example.com",
            "tom@example.com",
            "mary@example.com",
            "steve@example.com",
            "sarah@example.com",
            "mike@example.com",
            "lisa@example.com",
        ],
    }
    df = pd.DataFrame(data)
    return dd.from_pandas(df, npartitions=1)


def test_map_replace():
    df = create_test_dataframe()
    prompt = "Replace all personally identifiable terms with XX"
    map_op = dt.map(prompt=prompt)

    responses = [
        "{'first_name': 'XX', 'last_name': 'XX', 'email': 'XX'}",
        "{'first_name': 'XX', 'last_name': 'XX', 'email': 'XX'}",
        "{first_name: 'XX', last_name: 'XX'}",  # Invalid JSON - should cause error
        "{'first_name': 'XX', 'last_name': 'XX', 'email': 'XX'}",
        "{'first_name': 'XX', 'last_name': 'XX', 'email': 'XX'}",
        "{'first_name': 'XX', 'last_name': 'XX', 'email': 'XX'}",
        "{'first_name': 'XX', 'last_name': 'XX', 'email': 'XX'}",
        "{'first_name': 'XX', 'last_name': 'XX', 'email': 'XX'}",
        "{'first_name': 'XX', 'last_name': 'XX', 'email': 'XX'}",
        "{'first_name': 'XX', 'last_name': 'XX', 'email': 'XX'",  # Missing closing brace - should cause error
    ]

    expected_error_indices = [2, 9]
    mock_llm = MockLLM(responses)

    mapped = map_op(mock_llm, df)
    mapped = mapped.compute()

    for idx in range(len(mapped)):
        if idx in expected_error_indices:
            assert (
                mapped.loc[idx, ERRORED_COLUMN] == True
            ), f"Index {idx} should be marked as errored"
        else:
            assert (
                mapped.loc[idx, ERRORED_COLUMN] == False
            ), f"Index {idx} should not be marked as errored"

    mapped.to_csv("tests/test_data/test_map_replace_error_results.csv", index=False)
    return mapped


def create_test_filter_dataframe():
    data = {
        "statement": [
            "The Earth orbits the Sun.",
            "Pluto is the largest planet in our solar system.",
            "Water boils at 100 degrees Celsius at sea level.",
            "Vaccines cause autism.",
            "The Great Wall of China is visible from space.",
            "The capital of Australia is Sydney.",
            "Humans and dinosaurs lived at the same time.",
            "The chemical symbol for gold is Au.",
            "The human body has 206 bones.",
            "Mount Everest is in Africa.",
        ]
    }
    df = pd.DataFrame(data)
    return dd.from_pandas(df, npartitions=1)


def test_filter():
    df = create_test_filter_dataframe()
    prompt = "Check if the statement is factually correct."
    filter_op = dt.filter(prompt=prompt)

    # Based on the error, filter responses need to be in JSON format with curly braces
    responses = [
        '{"result": "TRUE"}',
        '{"result": "FALSE"}',
        "I think it's true",  # No JSON braces - should cause error
        '{"result": "TRUE"}',
        "Maybe",  # No JSON braces - should cause error
        '{"result": "FALSE"}',
        "The statement is false",  # No JSON braces - should cause error
        '{"result": "TRUE"}',
        '{"result": "TRUE"}',
        "UNTRUE",  # No JSON braces - should cause error
    ]

    expected_error_indices = [2, 4, 6, 9]

    mock_llm = MockLLM(responses)

    filtered = filter_op(mock_llm, df)
    filtered_df = filtered.compute()

    for idx in range(len(filtered_df)):
        if idx in expected_error_indices:
            assert (
                filtered_df.loc[idx, ERRORED_COLUMN] == True
            ), f"Index {idx} should be marked as errored"
        else:
            assert (
                filtered_df.loc[idx, ERRORED_COLUMN] == False
            ), f"Index {idx} should not be marked as errored"

    false_indices = [1, 5]
    for idx in range(len(filtered_df)):
        if idx in false_indices:
            assert (
                filtered_df.loc[idx, DELETED_COLUMN] == True
            ), f"Index {idx} (FALSE response) should be marked as deleted"
        elif idx not in expected_error_indices:
            assert (
                filtered_df.loc[idx, DELETED_COLUMN] == False
            ), f"Index {idx} (TRUE response) should not be marked as deleted"

    filtered_df.to_csv("tests/test_data/test_filter_error_results.csv", index=False)
    return filtered_df


def test_filter_on_error_delete():
    df = create_test_filter_dataframe()
    prompt = "Check if the statement is factually correct."
    filter_op = dt.filter(prompt=prompt, on_error="delete")

    # Using JSON format for valid responses, non-JSON for error cases
    responses = [
        '{"result": "TRUE"}',
        '{"result": "FALSE"}',
        "Not sure",  # No JSON braces - should cause error
        '{"result": "TRUE"}',
        "This is incorrect",  # No JSON braces - should cause error
        '{"result": "FALSE"}',
        "50% accurate",  # No JSON braces - should cause error
        '{"result": "TRUE"}',
        '{"result": "TRUE"}',
        Exception("Rate limit exceeded"),
    ]

    expected_error_indices = [2, 4, 6, 9]
    false_indices = [1, 5]

    mock_llm = MockLLM(responses)

    filtered = filter_op(mock_llm, df)
    filtered_df = filtered.compute()

    for idx in range(len(filtered_df)):
        if idx in expected_error_indices:
            assert (
                filtered_df.loc[idx, ERRORED_COLUMN] == True
            ), f"Index {idx} should be marked as errored"
        else:
            assert (
                filtered_df.loc[idx, ERRORED_COLUMN] == False
            ), f"Index {idx} should not be marked as errored"

    for idx in range(len(filtered_df)):
        if idx in false_indices or idx in expected_error_indices:
            assert (
                filtered_df.loc[idx, DELETED_COLUMN] == True
            ), f"Index {idx} should be marked as deleted"
        else:
            assert (
                filtered_df.loc[idx, DELETED_COLUMN] == False
            ), f"Index {idx} should not be marked as deleted"

    filtered_df.to_csv(
        "tests/test_data/test_filter_on_error_delete_results.csv", index=False
    )
    return filtered_df
