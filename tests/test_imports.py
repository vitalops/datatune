def test_imports():
    from datatune import dataset
    from datatune.apps import BaseApp
    
    ds = dataset()
    app = BaseApp(ds)
    
    assert app.dataset == ds