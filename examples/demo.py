import datatune as dt
import pandas as pd

# Initialize the workspace with URI and token
workspace = dt.Workspace(uri="dt://user_name/workspace_name", token="xyz")

dataset1 = workspace.add_dataset(name="dataset1", path="/data/dataset1.parquet", is_local=True)

# Set cloud credentials for accessing S3
workspace.add_credentials({
    'aws_access_key_id': 'YOUR_AWS_ACCESS_KEY_ID',
    'aws_secret_access_key': 'YOUR_AWS_SECRET_ACCESS_KEY',
    'region_name': 'YOUR_REGION'
})

# Assuming dataset2 is from S3
dataset2 = workspace.add_dataset(name="dataset2", path="s3://your-bucket-name/data/dataset2.parquet", is_local=False)

# List datasets initially available in the workspace
initial_datasets = workspace.list_datasets()
print("Initial datasets in the workspace:", [ds.name for ds in initial_datasets])

# Create a new view
view = workspace.create_view("view1")

# Extend the view with slices from datasets
view.extend(dataset_name="dataset1", slice_range=(20, 100))
view.extend(dataset_name="dataset2", slice_range=(30, 40))

# Add a new column to the view using DataFrame
df = pd.DataFrame({'new_column': [1, 2, 3]})
view.add_columns(data=df)

# Add columns manually without DataFrame
view.add_columns(column_name="manual_column", column_type="integer", default_value=0)
view.add_columns(column_name="manual_text", column_type="string", default_value="sample text")
print("Preview of view data:", view.display())

# Execute a query on the view
sql_query = "SELECT new_column, manual_column, manual_text FROM view1 WHERE new_column > 1"
queried_view = view.query(sql_query)
print("Preview of queried_view data:", queried_view.display())


# streaming use cases
def train_model_on_batch():
    pass


queried_view = queried_view.convert_to_pytorch()
ts = dt.TorchStream(queried_view)
for batch in ts.stream_batches():
    train_model_on_batch(batch)


# List the views available in the workspace
views = workspace.list_views()
print("Views in the workspace:", [v.name for v in views])

# Optionally, delete a dataset if needed
workspace.delete_dataset("dataset2")

# Optionally, delete a view if needed
workspace.delete_view("view1")

# Re-list datasets and views to see the changes
updated_datasets = workspace.list_datasets()
updated_views = workspace.list_views()
print("Updated datasets in the workspace:", [ds.name for ds in updated_datasets])
print("Updated views in the workspace:", [v.name for v in updated_views])
