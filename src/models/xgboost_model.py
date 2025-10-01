
"""
XGBoost Gradient Boosting Models
Max Depth 5 and Max Depth 10 implementations
"""

import os
import numpy as np
import xgboost as xgb
from typing import Optional
import joblib


class XGBoostModel:
    """XGBoost model for options pricing"""
    
    def __init__(self, max_depth: int = 10, name: Optional[str] = None):
        """
        Initialize XGBoost model
        
        Args:
            max_depth: Maximum tree depth
            name: Model name
        """
        self.max_depth = max_depth
        self.name = name or f"XGBoost_Max_Depth_{max_depth}"
        self.model = None
        self.is_trained = False
    
    def build_model(self) -> xgb.XGBRegressor:
        """
        Build XGBoost regressor
        
        Returns:
            XGBoost regressor
        """
        model = xgb.XGBRegressor(
            max_depth=self.max_depth,
            learning_rate=0.1,
            n_estimators=1000,
            objective='reg:squarederror',
            booster='gbtree',
            n_jobs=-1,
            gamma=0,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0,
            reg_lambda=1,
            random_state=42,
            early_stopping_rounds=50
        )
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> dict:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Training results
        """
        if self.model is None:
            self.model = self.build_model()
        
        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        self.is_trained = True
        
        # Get training results
        results = {
            'best_iteration': self.model.best_iteration,
            'best_score': self.model.best_score
        }
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict option prices
        
        Args:
            X: Input features
            
        Returns:
            Predicted prices
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        predictions = self.model.predict(X)
        return predictions
    
    def get_feature_importance(self) -> dict:
        """
        Get feature importance scores
        
        Returns:
            Dictionary of feature importances
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        return self.model.get_booster().get_score(importance_type='weight')
    
    def save(self, filepath: str):
        """Save model to file"""
        if self.model is not None:
            joblib.dump(self.model, filepath)
    
    def load(self, filepath: str):
        """Load model from file"""
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)
            self.is_trained = True
        else:
            raise FileNotFoundError(f"Model file not found: {filepath}")


class XGBoostMaxDepth5(XGBoostModel):
    """XGBoost model with max depth 5"""
    
    def __init__(self):
        super().__init__(max_depth=5, name="XGBoost_Max_Depth_5")


class XGBoostMaxDepth10(XGBoostModel):
    """XGBoost model with max depth 10 (best performing model)"""
    
    def __init__(self):
        super().__init__(max_depth=10, name="XGBoost_Max_Depth_10")
