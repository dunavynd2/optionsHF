"""Basic smoke tests for model imports and initialization"""
import sys
sys.path.insert(0, '/home/ubuntu/github_repos/optionsHF')

def test_imports():
    """Test that all modules can be imported"""
    from src.models.black_scholes import BlackScholesModel
    from src.models.neural_network import ThreeLayerFFNN, FiveLayerFFNN
    from src.models.xgboost_model import XGBoostMaxDepth5, XGBoostMaxDepth10
    from src.ensemble import ModelEnsemble
    print("✓ All imports successful")

def test_black_scholes():
    """Test Black-Scholes model"""
    from src.models.black_scholes import BlackScholesModel
    import numpy as np
    
    model = BlackScholesModel()
    
    # Test call option
    call_price = model.price_call(S=100, K=100, T=1.0, r=0.05, sigma=0.3)
    assert call_price > 0, "Call price should be positive"
    print(f"✓ Black-Scholes call price: ${call_price:.2f}")
    
    # Test put option
    put_price = model.price_put(S=100, K=100, T=1.0, r=0.05, sigma=0.3)
    assert put_price > 0, "Put price should be positive"
    print(f"✓ Black-Scholes put price: ${put_price:.2f}")

def test_ensemble_init():
    """Test ensemble initialization"""
    from src.ensemble import ModelEnsemble
    
    ensemble = ModelEnsemble(use_automl=False)
    assert len(ensemble.models) == 5, "Should have 5 models (excluding AutoML)"
    print(f"✓ Ensemble initialized with {len(ensemble.models)} models")

if __name__ == '__main__':
    test_imports()
    test_black_scholes()
    test_ensemble_init()
    print("\n✅ All tests passed!")
