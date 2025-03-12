def test_package_imports():
    """Test that all imports work properly without circular import errors."""
    # Import the main package and its components
    from datatune import Dataset, dataset
    from datatune.apps import BaseApp
    
    # Create a dataset and app to verify the imports work correctly
    ds = Dataset({"sample": "data"})
    app = BaseApp(ds)
    
    # Simple assertions to confirm everything loaded correctly
    assert hasattr(ds, "data")
    assert app.dataset == ds