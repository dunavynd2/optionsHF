
"""
Model Ensemble System
Combines predictions from all models with confidence scoring
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
        
        # Initialize models
        self._initialize_models()
        
        # Calculate inverse weights (lower MAE = higher weight)
        self._calculate_weights()
    
    def _initialize_models(self):
        """Initialize all models"""
        # Black-Scholes (always available)
        self.models['Black_Scholes'] = BlackScholesModel()
        
        # Neural Networks (require training or pre-trained weights)
        self.models['3_Layer_FFNN'] = ThreeLayerFFNN()
        self.models['5_Layer_FFNN'] = FiveLayerFFNN()
        
        # XGBoost models (require training or pre-trained weights)
        self.models['XGBoost_Max_Depth_5'] = XGBoostMaxDepth5()
        self.models['XGBoost_Max_Depth_10'] = XGBoostMaxDepth10()
        
        # AutoML (optional, requires credentials)
        if self.use_automl:
            try:
                self.models['AutoML_Regressor'] = AutoMLModel()
            except Exception as e:
                print(f"AutoML initialization failed: {e}")
                self.use_automl = False
    
    def _calculate_weights(self):
        """Calculate normalized weights based on inverse MAE"""
        # Inverse of MAE (lower error = higher weight)
        inverse_weights = {name: 1.0 / mae for name, mae in self.MODEL_WEIGHTS.items()}
        
        # Normalize weights to sum to 1
        total = sum(inverse_weights.values())
        self.normalized_weights = {name: w / total for name, w in inverse_weights.items()}
    
    def prepare_features_for_model(self, features_df: pd.DataFrame, model_name: str) -> np.ndarray:
        """
        Prepare features for specific model
        
        Args:
            features_df: DataFrame with raw features
            model_name: Name of the model
            
        Returns:
            Prepared feature array
        """
        # Core features used by all models
        feature_columns = [
            'underlying_price',
            'strike_price',
            'time_to_expiration',
            'risk_free_rate',
            'dividend_yield',
            'moneyness',
            'is_call'
        ]
        
        # Add model-specific features
        if model_name in ['XGBoost_Max_Depth_5', 'XGBoost_Max_Depth_10']:
            # XGBoost can handle more features
            feature_columns.extend(['volume', 'open_interest', 'relative_spread'])
        
        # Select available features
        available_features = [col for col in feature_columns if col in features_df.columns]
        
        return features_df[available_features].values
    
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
                    'risk_free_rate': row['risk_free_rate'],
                    'volatility': volatility,
                    'dividend_yield': row.get('dividend_yield', 0.0),
                    'is_call': row.get('is_call', 1)
                }
                predictions.append(model.predict(features_dict))
            return np.array(predictions)
        
        # Other models
        X = self.prepare_features_for_model(features_df, model_name)
        
        if not model.is_trained:
            # Return zeros if model not trained (placeholder)
            return np.zeros(len(features_df))
        
        return model.predict(X)
    
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
            ensemble_pred = all_predictions.get('XGBoost_Max_Depth_10', 
                                               np.zeros(len(features_df)))
        
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        # Calculate confidence score based on model agreement
        confidence_scores = self._calculate_confidence(all_predictions)
        
        metadata = {
            'individual_predictions': all_predictions,
            'confidence_scores': confidence_scores,
            'method': method,
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
        
        # Stack predictions
        pred_array = np.array(list(all_predictions.values()))
        
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
        
        # Calculate statistics
        pred_array = np.array(list(individual_preds.values()))
        
        analysis = {
            'ensemble_prediction': ensemble_pred.tolist(),
            'individual_predictions': {k: v.tolist() for k, v in individual_preds.items()},
            'confidence_scores': metadata['confidence_scores'].tolist(),
            'statistics': {
                'mean': np.mean(pred_array, axis=0).tolist(),
                'median': np.median(pred_array, axis=0).tolist(),
                'std': np.std(pred_array, axis=0).tolist(),
                'min': np.min(pred_array, axis=0).tolist(),
                'max': np.max(pred_array, axis=0).tolist()
            },
            'model_weights': self.normalized_weights,
            'timestamp': metadata['timestamp']
        }
        
        return analysis
