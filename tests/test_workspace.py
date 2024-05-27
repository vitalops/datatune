import unittest
from unittest.mock import patch, MagicMock
from datatune import Workspace
from datatune.exceptions import DatatuneException

class TestWorkspace(unittest.TestCase):
    def setUp(self):
        self.uri = "dt://user_name/workspace_name"
        self.token = "xyz"
        self.workspace = Workspace(self.uri, self.token)

    @patch('datatune.api.API.delete')
    @patch('datatune.api.API.put')
    @patch('datatune.api.API.get')
    @patch('datatune.api.API.post')
    def test_workspace_operations(self, mock_post, mock_get, mock_put, mock_delete):
        # Mocking API responses
        mock_post.side_effect = [
            {'success': True},  # for add_dataset "dataset1"
            {'success': True},  # for add_dataset "dataset2"
            {'success': True},  # for create_view "view1"
            {'success': True},  # for add_columns
            {'success': True},  # for add_filter
            {'success': True},  # for sort_by
            {'success': True}   # for group_by
        ]
        mock_put.return_value = {'success': True}  # for extend
        mock_get.side_effect = [
            {'success': True, 'datasets': ['dataset1', 'dataset2']},  # initial list_datasets
            {'success': True, 'views': ['view1']},  # initial list_views
            {'success': True, 'datasets': ['dataset1']},  # final list_datasets after delete
            {'success': True, 'views': []}  # final list_views after delete
        ]
        mock_delete.return_value = {'success': True}

        # Testing dataset addition
        self.workspace.add_dataset("dataset1", "/data/dataset1.parquet")
        self.workspace.add_dataset("dataset2", "/data/dataset2.parquet")
        datasets = self.workspace.list_datasets()
        self.assertIn("dataset1", datasets)
        self.assertIn("dataset2", datasets)

        # Testing view creation and modification
        view = self.workspace.create_view("view1")
        view.extend("dataset1", (20, 100))
        view.extend("dataset2", (30, 40))
        view.add_columns("new_column", "integer", 0)
        view.add_filter("score", ">=0.5")
        view.sort_by("score", "descending")
        view.group_by("label")

        views = self.workspace.list_views()
        self.assertIn("view1", views)

        # Deleting dataset and view
        self.workspace.delete_dataset("dataset2")
        self.workspace.delete_view("view1")

        # Verifying final state
        updated_datasets = self.workspace.list_datasets()
        updated_views = self.workspace.list_views()
        self.assertIn("dataset1", updated_datasets)
        self.assertNotIn("dataset2", updated_datasets)
        self.assertFalse(updated_views)

if __name__ == '__main__':
    unittest.main()
