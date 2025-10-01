
"""
Feed-Forward Neural Network Models
3-Layer and 5-Layer FFNN implementations using TensorFlow
Loads pre-trained models from file paths
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from typing import Optional, List


class FeedForwardNN:
    """Feed-Forward Neural Network for options pricing - loads pre-trained models"""
    
    def __init__(self, layers: List[int], model_path: Optional[str] = None, name: str = "FFNN"):
        """
        Initialize FFNN model
        
        Args:
            layers: List of hidden layer sizes (for reference)
            model_path: Path to pre-trained model file
            name: Model name
        """
        self.layers = layers
        self.name = name
        self.model_path = model_path
        self.model = None
        self.is_trained = False
        
        # Load model if path is provided
        if model_path:
            self.load(model_path)
    
    def load(self, filepath: str):
        """
        Load pre-trained TensorFlow/Keras model from file
        
        Args:
            filepath: Path to the model file (.h5, .keras, or SavedModel directory)
            
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
            # Load Keras model
            self.model = keras.models.load_model(filepath, compile=False)
            
            # Recompile with appropriate loss and metrics
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mean_absolute_error',
                metrics=['mean_absolute_percentage_error']
            )
            
            self.is_trained = True
            print(f"Successfully loaded pre-trained model from: {filepath}")
        except Exception as e:
            raise Exception(f"Failed to load model from {filepath}: {str(e)}")
    
    def prepare_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for neural network prediction
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Numpy array with features
        """
        # Neural networks typically use the same 27 features as XGBoost
        # Ensure features are in the correct format
        if isinstance(features_df, pd.DataFrame):
            return features_df.values
        return features_df
    
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predict option prices using pre-trained model
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Array of predicted prices
            
        Raises:
            ValueError: If model not loaded
        """
        if self.model is None:
            raise ValueError(
                f"Model not loaded. Please load a pre-trained model using load() method.\n"
                f"Expected model path: {self.model_path or 'Not specified'}"
            )
        
        # Prepare features
        X = self.prepare_features(features_df)
        
        # Make predictions
        predictions = self.model.predict(X, verbose=0)
        
        return predictions.flatten()
    
    def get_model_summary(self) -> str:
        """
        Get model architecture summary
        
        Returns:
            String representation of model architecture
            
        Raises:
            ValueError: If model not loaded
        """
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        # Capture model summary
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        return "\n".join(stringlist)


class ThreeLayerFFNN(FeedForwardNN):
    """3-Layer Feed-Forward Neural Network (MAE: 8.8075)"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize 3-Layer FFNN model
        
        Args:
            model_path: Path to pre-trained model file
                       If not provided, uses FFNN_3_LAYER_MODEL_PATH env variable
        """
        if model_path is None:
            model_path = os.getenv('FFNN_3_LAYER_MODEL_PATH')
        
        super().__init__(layers=[128, 64, 32], model_path=model_path, name="3_Layer_FFNN")


class FiveLayerFFNN(FeedForwardNN):
    """5-Layer Feed-Forward Neural Network (MAE: 4.6374)"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize 5-Layer FFNN model
        
        Args:
            model_path: Path to pre-trained model file
                       If not provided, uses FFNN_5_LAYER_MODEL_PATH env variable
        """
        if model_path is None:
            model_path = os.getenv('FFNN_5_LAYER_MODEL_PATH')
        
        super().__init__(layers=[256, 128, 64, 32, 16], model_path=model_path, name="5_Layer_FFNN")
