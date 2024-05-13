import datatune as dt
import pandas as pd

# Initialize the workspace with URI and token
workspace = dt.Workspace(uri="dt://user_name/workspace_name", token="xyz")

# Add local dataset
dataset1 = workspace.add_dataset(name="dataset1", path="/data/dataset1.parquet", is_local=True)

# Set cloud credentials for accessing S3
workspace.add_credentials({
    'aws_access_key_id': 'YOUR_AWS_ACCESS_KEY_ID',
    'aws_secret_access_key': 'YOUR_AWS_SECRET_ACCESS_KEY',
    'region_name': 'YOUR_REGION'
})

# Add remote dataset from S3
dataset2 = workspace.add_dataset(name="dataset2", path="s3://your-bucket-name/data/dataset2.parquet", is_local=False)

# List initial datasets in the workspace
initial_datasets = workspace.list_datasets()
print("Initial datasets in the workspace:", [ds.name for ds in initial_datasets])

# Create a new view
view = workspace.create_view("view1")

# Extend the view with slices from datasets
view.extend(dataset_name="dataset1", slice_range=(20, 100))
view.extend(dataset_name="dataset2", slice_range=(30, 40))

# Add a new column using DataFrame
df = pd.DataFrame({'new_column': [1, 2, 3]})
view.add_columns(data=df)

# Add columns manually without DataFrame
view.add_columns(column_name="manual_column", column_type="integer", default_value=0)
view.add_columns(column_name="manual_text", column_type="string", default_value="sample text")

# Display the view using filter and sort functionality from Query class
filtered_view = view.filter("new_column > 1")
print("Preview of view data:", filtered_view.display())

def train(batch):
    pass

stream = dt.Stream(source=filtered_view)
batches = stream.stream_batches()

for batch in batches:
    train(batch)


# List available views in the workspace
views = workspace.list_views()
print("Views in the workspace:", [v.name for v in views])

# Delete a dataset if needed
workspace.delete_dataset("dataset2")

# Delete a view if needed
workspace.delete_view("view1")

# Re-list datasets and views to see the changes
updated_datasets = workspace.list_datasets()
updated_views = workspace.list_views()
print("Updated datasets in the workplace:", [ds.name for ds in updated_datasets])
print("Updated views in the workplace:", [v.name for v in updated_views])
