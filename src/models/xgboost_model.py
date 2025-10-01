
"""
XGBoost Gradient Boosting Models
Max Depth 5 and Max Depth 10 implementations
Loads pre-trained models from file paths
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Optional, List


class XGBoostModel:
    """XGBoost model for options pricing - loads pre-trained models"""
    
    # Exact feature names required by the pre-trained models (27 features total)
    FEATURE_NAMES = [
        'strike_price',
        'implied_volatility',
        'zero_coupon_rate',
        'index_dividend_yields',
        'option_type',  # 1 for call, 0 for put
        'time_to_maturity',
        'underlying_asset_current_price',
        'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
        'lag_6', 'lag_7', 'lag_8', 'lag_9', 'lag_10',
        'lag_11', 'lag_12', 'lag_13', 'lag_14', 'lag_15',
        'lag_16', 'lag_17', 'lag_18', 'lag_19', 'lag_20'
    ]
    
    def __init__(self, max_depth: int = 10, model_path: Optional[str] = None, name: Optional[str] = None):
        """
        Initialize XGBoost model
        
        Args:
            max_depth: Maximum tree depth (5 or 10)
            model_path: Path to pre-trained model file
            name: Model name
        """
        self.max_depth = max_depth
        self.name = name or f"XGBoost_Max_Depth_{max_depth}"
        self.model_path = model_path
        self.model = None
        self.is_trained = False
        
        # Load model if path is provided
        if model_path:
            self.load(model_path)
    
    def load(self, filepath: str):
        """
        Load pre-trained XGBoost model from file
        
        Args:
            filepath: Path to the model file
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Model file not found: {filepath}\n"
                f"Please ensure the pre-trained model file is uploaded to this location.\n"
                f"See the documentation for instructions on uploading model files."
            )
        
        try:
            # Load model using xgb.Booster() as specified
            self.model = xgb.Booster()
            self.model.load_model(filepath)
            self.is_trained = True
            print(f"Successfully loaded pre-trained model from: {filepath}")
        except Exception as e:
            raise Exception(f"Failed to load model from {filepath}: {str(e)}")
    
    def prepare_features(self, features_df: pd.DataFrame) -> xgb.DMatrix:
        """
        Prepare features in the exact format required by the model
        
        Args:
            features_df: DataFrame with raw features
            
        Returns:
            xgb.DMatrix with properly formatted features
            
        Raises:
            ValueError: If required features are missing
        """
        # Check for required features
        missing_features = [f for f in self.FEATURE_NAMES if f not in features_df.columns]
        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}\n"
                f"Required features: {self.FEATURE_NAMES}"
            )
        
        # Select features in the exact order
        X = features_df[self.FEATURE_NAMES].copy()
        
        # Convert to DMatrix format as required
        dmatrix = xgb.DMatrix(X, feature_names=self.FEATURE_NAMES)
        
        return dmatrix
    
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predict option prices using pre-trained model
        
        Args:
            features_df: DataFrame with features (must contain all 27 required features)
            
        Returns:
            Array of predicted prices
            
        Raises:
            ValueError: If model not loaded or features are invalid
        """
        if self.model is None:
            raise ValueError(
                f"Model not loaded. Please load a pre-trained model using load() method.\n"
                f"Expected model path: {self.model_path or 'Not specified'}"
            )
        
        # Prepare features in DMatrix format
        dmatrix = self.prepare_features(features_df)
        
        # Make predictions
        predictions = self.model.predict(dmatrix)
        
        return predictions
    
    def get_feature_importance(self) -> dict:
        """
        Get feature importance scores from the loaded model
        
        Returns:
            Dictionary of feature importances
            
        Raises:
            ValueError: If model not loaded
        """
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        return self.model.get_score(importance_type='weight')
    
    @staticmethod
    def validate_features(features_df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that all required features are present
        
        Args:
            features_df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, missing_features)
        """
        missing = [f for f in XGBoostModel.FEATURE_NAMES if f not in features_df.columns]
        return len(missing) == 0, missing


class XGBoostMaxDepth5(XGBoostModel):
    """XGBoost model with max depth 5 (MAE: 1.6362)"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize XGBoost Max Depth 5 model
        
        Args:
            model_path: Path to pre-trained model file
                       If not provided, uses XGBOOST_DEPTH_5_MODEL_PATH env variable
        """
        if model_path is None:
            model_path = os.getenv('XGBOOST_DEPTH_5_MODEL_PATH')
        
        super().__init__(max_depth=5, model_path=model_path, name="XGBoost_Max_Depth_5")


class XGBoostMaxDepth10(XGBoostModel):
    """XGBoost model with max depth 10 - Best performing model (MAE: 0.8093)"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize XGBoost Max Depth 10 model
        
        Args:
            model_path: Path to pre-trained model file
                       If not provided, uses XGBOOST_DEPTH_10_MODEL_PATH env variable
        """
        if model_path is None:
            model_path = os.getenv('XGBOOST_DEPTH_10_MODEL_PATH')
        
        super().__init__(max_depth=10, model_path=model_path, name="XGBoost_Max_Depth_10")
