"""Minimal smoke tests without TensorFlow"""
import sys
sys.path.insert(0, '/home/ubuntu/github_repos/optionsHF')

def test_black_scholes():
    """Test Black-Scholes model"""
    from src.models.black_scholes import BlackScholesModel
    
    model = BlackScholesModel()
    call_price = model.price_call(S=100, K=100, T=1.0, r=0.05, sigma=0.3)
    assert call_price > 0
    print(f"✓ Black-Scholes works: call=${call_price:.2f}")

def test_xgboost():
    """Test XGBoost model initialization"""
    from src.models.xgboost_model import XGBoostMaxDepth10
    
    model = XGBoostMaxDepth10()
    assert model.max_depth == 10
    print(f"✓ XGBoost model initialized")

if __name__ == '__main__':
    test_black_scholes()
    test_xgboost()
    print("\n✅ Core models validated!")
