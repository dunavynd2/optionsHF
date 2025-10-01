
"""
Model Ensemble System
Combines predictions from all models with confidence scoring
Updated to work with pre-trained models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from src.models.black_scholes import BlackScholesModel
from src.models.neural_network import ThreeLayerFFNN, FiveLayerFFNN
from src.models.xgboost_model import XGBoostMaxDepth5, XGBoostMaxDepth10
from src.models.automl import AutoMLModel


class ModelEnsemble:
    """Ensemble of all pricing models"""
    
    # Model performance metrics from research (MAE)
    MODEL_WEIGHTS = {
        'XGBoost_Max_Depth_10': 0.8093,    # Best model
        'AutoML_Regressor': 1.0248,
        'XGBoost_Max_Depth_5': 1.6362,
        '5_Layer_FFNN': 4.6374,
        '3_Layer_FFNN': 8.8075,
        'Black_Scholes': 8.0082
    }
    
    def __init__(self, use_automl: bool = False):
        """
        Initialize model ensemble
        
        Args:
            use_automl: Whether to include AutoML in predictions (requires credentials)
        """
        self.models = {}
        self.use_automl = use_automl
        self.model_load_status = {}
        
        # Initialize models
        self._initialize_models()
        
        # Calculate inverse weights (lower MAE = higher weight)
        self._calculate_weights()
    
    def _initialize_models(self):
        """Initialize all models - load pre-trained models where available"""
        # Black-Scholes (always available, no training needed)
        self.models['Black_Scholes'] = BlackScholesModel()
        self.model_load_status['Black_Scholes'] = True
        
        # XGBoost models (load pre-trained)
        try:
            self.models['XGBoost_Max_Depth_10'] = XGBoostMaxDepth10()
            self.model_load_status['XGBoost_Max_Depth_10'] = self.models['XGBoost_Max_Depth_10'].is_trained
            if not self.model_load_status['XGBoost_Max_Depth_10']:
                print("Warning: XGBoost Max Depth 10 model not loaded. Set XGBOOST_DEPTH_10_MODEL_PATH environment variable.")
        except Exception as e:
            print(f"XGBoost Max Depth 10 initialization failed: {e}")
            self.model_load_status['XGBoost_Max_Depth_10'] = False
        
        try:
            self.models['XGBoost_Max_Depth_5'] = XGBoostMaxDepth5()
            self.model_load_status['XGBoost_Max_Depth_5'] = self.models['XGBoost_Max_Depth_5'].is_trained
            if not self.model_load_status['XGBoost_Max_Depth_5']:
                print("Warning: XGBoost Max Depth 5 model not loaded. Set XGBOOST_DEPTH_5_MODEL_PATH environment variable.")
        except Exception as e:
            print(f"XGBoost Max Depth 5 initialization failed: {e}")
            self.model_load_status['XGBoost_Max_Depth_5'] = False
        
        # Neural Networks (load pre-trained)
        try:
            self.models['3_Layer_FFNN'] = ThreeLayerFFNN()
            self.model_load_status['3_Layer_FFNN'] = self.models['3_Layer_FFNN'].is_trained
            if not self.model_load_status['3_Layer_FFNN']:
                print("Warning: 3-Layer FFNN model not loaded. Set FFNN_3_LAYER_MODEL_PATH environment variable.")
        except Exception as e:
            print(f"3-Layer FFNN initialization failed: {e}")
            self.model_load_status['3_Layer_FFNN'] = False
        
        try:
            self.models['5_Layer_FFNN'] = FiveLayerFFNN()
            self.model_load_status['5_Layer_FFNN'] = self.models['5_Layer_FFNN'].is_trained
            if not self.model_load_status['5_Layer_FFNN']:
                print("Warning: 5-Layer FFNN model not loaded. Set FFNN_5_LAYER_MODEL_PATH environment variable.")
        except Exception as e:
            print(f"5-Layer FFNN initialization failed: {e}")
            self.model_load_status['5_Layer_FFNN'] = False
        
        # AutoML (optional, requires credentials)
        if self.use_automl:
            try:
                self.models['AutoML_Regressor'] = AutoMLModel()
                self.model_load_status['AutoML_Regressor'] = True
            except Exception as e:
                print(f"AutoML initialization failed: {e}")
                self.use_automl = False
                self.model_load_status['AutoML_Regressor'] = False
    
    def _calculate_weights(self):
        """Calculate normalized weights based on inverse MAE"""
        # Inverse of MAE (lower error = higher weight)
        inverse_weights = {name: 1.0 / mae for name, mae in self.MODEL_WEIGHTS.items()}
        
        # Normalize weights to sum to 1
        total = sum(inverse_weights.values())
        self.normalized_weights = {name: w / total for name, w in inverse_weights.items()}
    
    def prepare_features_for_xgboost(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features specifically for XGBoost models (27 features)
        
        Args:
            features_df: DataFrame with raw features from BigQuery
            
        Returns:
            DataFrame with 27 required features for XGBoost
        """
        # Map BigQuery features to XGBoost feature names
        xgb_features = pd.DataFrame()
        
        # Direct mappings
        xgb_features['strike_price'] = features_df['strike_price']
        xgb_features['zero_coupon_rate'] = features_df.get('risk_free_rate', 0.05)
        xgb_features['index_dividend_yields'] = features_df.get('dividend_yield', 0.0)
        xgb_features['option_type'] = features_df.get('is_call', 1)  # 1 for call, 0 for put
        xgb_features['time_to_maturity'] = features_df['time_to_expiration']
        xgb_features['underlying_asset_current_price'] = features_df['underlying_price']
        
        # Implied volatility - use relative_spread as proxy or default
        xgb_features['implied_volatility'] = features_df.get('relative_spread', 0.3) * 2
        xgb_features['implied_volatility'] = xgb_features['implied_volatility'].clip(0.1, 1.0)
        
        # Lag features (past 20 days' closing prices)
        # If not available in features_df, use current price as placeholder
        for i in range(1, 21):
            lag_col = f'lag_{i}'
            if lag_col in features_df.columns:
                xgb_features[lag_col] = features_df[lag_col]
            else:
                # Use current price as fallback (not ideal, but prevents errors)
                xgb_features[lag_col] = features_df['underlying_price']
        
        return xgb_features
    
    def prepare_features_for_model(self, features_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """
        Prepare features for specific model
        
        Args:
            features_df: DataFrame with raw features
            model_name: Name of the model
            
        Returns:
            Prepared feature DataFrame
        """
        if model_name in ['XGBoost_Max_Depth_5', 'XGBoost_Max_Depth_10']:
            return self.prepare_features_for_xgboost(features_df)
        
        # For neural networks, use the same features as XGBoost
        if model_name in ['3_Layer_FFNN', '5_Layer_FFNN']:
            return self.prepare_features_for_xgboost(features_df)
        
        # For other models, return as-is
        return features_df
    
    def predict_single_model(self, model_name: str, features_df: pd.DataFrame) -> np.ndarray:
        """
        Get predictions from a single model
        
        Args:
            model_name: Name of the model
            features_df: DataFrame with features
            
        Returns:
            Array of predictions
        """
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        # Check if model is loaded
        if not self.model_load_status.get(model_name, False):
            print(f"Warning: {model_name} not loaded, returning zeros")
            return np.zeros(len(features_df))
        
        # Black-Scholes requires special handling
        if model_name == 'Black_Scholes':
            predictions = []
            for _, row in features_df.iterrows():
                # Estimate volatility from relative spread or use default
                volatility = row.get('relative_spread', 0.3) * 2  # Simple proxy
                volatility = max(0.1, min(volatility, 1.0))  # Clamp between 0.1 and 1.0
                
                features_dict = {
                    'underlying_price': row['underlying_price'],
                    'strike_price': row['strike_price'],
                    'time_to_expiration': row['time_to_expiration'],
                    'risk_free_rate': row.get('risk_free_rate', 0.05),
                    'volatility': volatility,
                    'dividend_yield': row.get('dividend_yield', 0.0),
                    'is_call': row.get('is_call', 1)
                }
                predictions.append(model.predict(features_dict))
            return np.array(predictions)
        
        # Prepare features for the specific model
        prepared_features = self.prepare_features_for_model(features_df, model_name)
        
        # Get predictions
        try:
            return model.predict(prepared_features)
        except Exception as e:
            print(f"Error predicting with {model_name}: {e}")
            return np.zeros(len(features_df))
    
    def predict_all_models(self, features_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get predictions from all models
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Dictionary of model predictions
        """
        predictions = {}
        
        for model_name in self.models.keys():
            if model_name == 'AutoML_Regressor' and not self.use_automl:
                continue
            
            try:
                predictions[model_name] = self.predict_single_model(model_name, features_df)
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
                predictions[model_name] = np.zeros(len(features_df))
        
        return predictions
    
    def predict_ensemble(self, features_df: pd.DataFrame, 
                        method: str = 'weighted_average') -> Tuple[np.ndarray, Dict]:
        """
        Get ensemble predictions
        
        Args:
            features_df: DataFrame with features
            method: Ensemble method ('weighted_average', 'median', 'best_model')
            
        Returns:
            Tuple of (predictions, metadata)
        """
        # Get all model predictions
        all_predictions = self.predict_all_models(features_df)
        
        if method == 'weighted_average':
            # Weighted average based on model performance
            ensemble_pred = np.zeros(len(features_df))
            total_weight = 0
            
            for model_name, predictions in all_predictions.items():
                if self.model_load_status.get(model_name, False):
                    weight = self.normalized_weights.get(model_name, 0)
                    ensemble_pred += weight * predictions
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_pred /= total_weight
        
        elif method == 'median':
            # Median of all predictions
            pred_array = np.array(list(all_predictions.values()))
            ensemble_pred = np.median(pred_array, axis=0)
        
        elif method == 'best_model':
            # Use only the best model (XGBoost Max Depth 10)
            if self.model_load_status.get('XGBoost_Max_Depth_10', False):
                ensemble_pred = all_predictions.get('XGBoost_Max_Depth_10', 
                                                   np.zeros(len(features_df)))
            else:
                print("Warning: Best model (XGBoost Max Depth 10) not loaded, using Black-Scholes")
                ensemble_pred = all_predictions.get('Black_Scholes', 
                                                   np.zeros(len(features_df)))
        
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        # Calculate confidence score based on model agreement
        confidence_scores = self._calculate_confidence(all_predictions)
        
        metadata = {
            'individual_predictions': all_predictions,
            'confidence_scores': confidence_scores,
            'method': method,
            'model_load_status': self.model_load_status,
            'timestamp': datetime.now().isoformat()
        }
        
        return ensemble_pred, metadata
    
    def _calculate_confidence(self, all_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate confidence scores based on model agreement
        
        Args:
            all_predictions: Dictionary of model predictions
            
        Returns:
            Array of confidence scores (0-1)
        """
        if not all_predictions:
            return np.array([0.0])
        
        # Only use predictions from loaded models
        loaded_predictions = [pred for name, pred in all_predictions.items() 
                            if self.model_load_status.get(name, False)]
        
        if not loaded_predictions:
            return np.array([0.0] * len(next(iter(all_predictions.values()))))
        
        # Stack predictions
        pred_array = np.array(loaded_predictions)
        
        # Calculate coefficient of variation (lower = higher confidence)
        mean_pred = np.mean(pred_array, axis=0)
        std_pred = np.std(pred_array, axis=0)
        
        # Avoid division by zero
        cv = np.where(mean_pred > 0, std_pred / mean_pred, 1.0)
        
        # Convert to confidence score (0-1, higher is better)
        confidence = 1.0 / (1.0 + cv)
        
        return confidence
    
    def analyze_option(self, features_df: pd.DataFrame) -> Dict:
        """
        Comprehensive option analysis with all models
        
        Args:
            features_df: DataFrame with option features
            
        Returns:
            Dictionary with analysis results
        """
        # Get ensemble prediction
        ensemble_pred, metadata = self.predict_ensemble(features_df, method='weighted_average')
        
        # Get individual model predictions
        individual_preds = metadata['individual_predictions']
        
        # Calculate statistics (only from loaded models)
        loaded_preds = [pred for name, pred in individual_preds.items() 
                       if self.model_load_status.get(name, False)]
        
        if loaded_preds:
            pred_array = np.array(loaded_preds)
            statistics = {
                'mean': np.mean(pred_array, axis=0).tolist(),
                'median': np.median(pred_array, axis=0).tolist(),
                'std': np.std(pred_array, axis=0).tolist(),
                'min': np.min(pred_array, axis=0).tolist(),
                'max': np.max(pred_array, axis=0).tolist()
            }
        else:
            statistics = {
                'mean': [0.0],
                'median': [0.0],
                'std': [0.0],
                'min': [0.0],
                'max': [0.0]
            }
        
        analysis = {
            'ensemble_prediction': ensemble_pred.tolist(),
            'individual_predictions': {k: v.tolist() for k, v in individual_preds.items()},
            'confidence_scores': metadata['confidence_scores'].tolist(),
            'statistics': statistics,
            'model_weights': self.normalized_weights,
            'model_load_status': self.model_load_status,
            'timestamp': metadata['timestamp']
        }
        
        return analysis
