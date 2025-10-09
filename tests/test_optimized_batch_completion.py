import itertools
import unittest
from unittest.mock import patch
from datatune.llm.llm import OpenAI
import itertools

class TestOptimizedBatchCompletion(unittest.TestCase):
    @patch("datatune.llm.llm.batch_completion")
    def test_cleaning_of_llm_output(self, mock_batch_completion):

        mock_batch_completion.return_value = [
            {
                "choices": [
                    {"message": {"content": "index=0|{'hello'}<endofrow>index=1|{'world'}<endofrow>"}}
                ]
            }
        ]

        llm = OpenAI()
        
        result = llm(
            ["row1", "row2"],
            optimized=True
        )
        self.assertEqual(len(result), 2)
        self.assertIn("hello", result[0])
        self.assertIn("world", result[1])
        self.assertTrue(all(isinstance(x, (str, Exception)) for x in result))

    @patch("datatune.llm.llm.batch_completion")
    def test_empty_input(self, mock_batch_completion):
        """Test that empty input returns an empty list."""
        llm = OpenAI()
        result = llm([], optimized=True)
        self.assertEqual(result, [])

    @patch("datatune.llm.llm.batch_completion")
    def test_retry_succeeds(self, mock_batch_completion):
        malformed_response = [
            {"choices": [{"message": {"content": "index=0|{bad_format}<endofrow>index=1|{malformed}"}}]}
        ]
        good_response = [
            {"choices": [{"message": {"content": "index=0|{'fixed'}<endofrow>index=1|{'fixed'}"}}]}
        ]

        mock_batch_completion.side_effect = [malformed_response, good_response]

        llm = OpenAI()

        result = llm(["row1", "row2"], max_retries=2, optimized=True)

        self.assertEqual(result, ["fixed", "fixed"])
        self.assertEqual(mock_batch_completion.call_count, 2)

    @patch("datatune.llm.llm.batch_completion")
    def test_exhaust_retries(self, mock_batch_completion):
        malformed_response = [
            {"choices": [{"message": {"content": "index=0|completely_unparseable"}}]}
        ]
        
        mock_batch_completion.side_effect = itertools.cycle([malformed_response])

        llm = OpenAI()

        result = llm(["row0"], max_retries=3, optimized=True)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], Exception)
