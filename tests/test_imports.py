def test_imports():
    try:
        from ioninsertml.bayesian_opt import bo
        from ioninsertml.utils import data_loader, generate_configs
        assert True
    except ImportError as e:
        assert False, f"Import failed: {e}"
