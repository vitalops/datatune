import datatune as dt

# Initialize the workspace with URI and token
workspace = dt.Workspace(uri="dt://user_name/workspace_name", token="xyz")

# Add datasets to the workspace
workspace.add_dataset(name="dataset1", path="/data/dataset1.parquet")
workspace.add_dataset(name="dataset2", path="/data/dataset2.parquet")

# List datasets initially available in the workspace
initial_datasets = workspace.list_datasets()
print("Initial datasets in the workspace:", initial_datasets)

# Create a new view
view = workspace.create_view("view1")

# Extend the view with slices from datasets
view.extend(dataset_name="dataset1", slice_range=(20, 100))
view.extend(dataset_name="dataset2", slice_range=(30, 40))

# Add a new column to the view
view.add_columns(column_name="new_column", column_type="integer", default_value=0)

# List the views available in the workspace
views = workspace.list_views()
print("Views in the workspace:", views)

# Optionally, delete a dataset if needed
workspace.delete_dataset("dataset2")

# Optionally, delete a view if needed
workspace.delete_view("view1")

# Re-list datasets and views to see the changes
updated_datasets = workspace.list_datasets()
updated_views = workspace.list_views()
print("Updated datasets in the workspace:", updated_datasets)
print("Updated views in the workspace:", updated_views)
