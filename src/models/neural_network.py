
"""
Feed-Forward Neural Network Models
3-Layer and 5-Layer FFNN implementations using TensorFlow
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Optional, List


class FeedForwardNN:
    """Feed-Forward Neural Network for options pricing"""
    
    def __init__(self, layers: List[int], name: str = "FFNN"):
        """
        Initialize FFNN model
        
        Args:
            layers: List of hidden layer sizes
            name: Model name
        """
        self.layers = layers
        self.name = name
        self.model = None
        self.is_trained = False
    
    def build_model(self, input_dim: int) -> keras.Model:
        """
        Build neural network architecture
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential(name=self.name)
        
        # Input layer
        model.add(keras.layers.Input(shape=(input_dim,)))
        
        # Hidden layers
        for i, units in enumerate(self.layers):
            model.add(keras.layers.Dense(
                units,
                activation='relu',
                kernel_initializer='he_normal',
                name=f'hidden_{i+1}'
            ))
            model.add(keras.layers.Dropout(0.2, name=f'dropout_{i+1}'))
        
        # Output layer
        model.add(keras.layers.Dense(1, activation='linear', name='output'))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_absolute_error',
            metrics=['mean_absolute_percentage_error']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 100, batch_size: int = 32) -> dict:
        """
        Train the neural network
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        if self.model is None:
            self.model = self.build_model(X_train.shape[1])
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train model
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        self.is_trained = True
        return history.history
    
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
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def save(self, filepath: str):
        """Save model to file"""
        if self.model is not None:
            self.model.save(filepath)
    
    def load(self, filepath: str):
        """Load model from file"""
        if os.path.exists(filepath):
            self.model = keras.models.load_model(filepath)
            self.is_trained = True
        else:
            raise FileNotFoundError(f"Model file not found: {filepath}")


class ThreeLayerFFNN(FeedForwardNN):
    """3-Layer Feed-Forward Neural Network"""
    
    def __init__(self):
        super().__init__(layers=[128, 64, 32], name="3_Layer_FFNN")


class FiveLayerFFNN(FeedForwardNN):
    """5-Layer Feed-Forward Neural Network"""
    
    def __init__(self):
        super().__init__(layers=[256, 128, 64, 32, 16], name="5_Layer_FFNN")
