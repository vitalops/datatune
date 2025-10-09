import unittest
from unittest.mock import patch, MagicMock
from datatune.llm.llm import OpenAI

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
        self.assertTrue(all(isinstance(x, str) for x in result))

    @patch("datatune.llm.llm.batch_completion")
    def test_empty_input(self, mock_batch_completion):
        """Test that empty input returns an empty list."""
        llm = OpenAI()
        result = llm([], optimized=True)
        self.assertEqual(result, [])
