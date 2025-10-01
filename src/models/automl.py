
"""
Google Cloud AutoML Integration
Wrapper for AutoML Regressor predictions
"""

import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from google.cloud import automl_v1beta1 as automl
from google.oauth2 import service_account


class AutoMLModel:
    """Google Cloud AutoML model wrapper"""
    
    def __init__(self, model_id: Optional[str] = None, project_id: Optional[str] = None):
        """
        Initialize AutoML model
        
        Args:
            model_id: AutoML model ID (default from env)
            project_id: Google Cloud project ID
        """
        self.model_id = model_id or os.getenv('AUTOML_MODEL_ID', '3224465183011241984')
        self.project_id = project_id or os.getenv('GOOGLE_CLOUD_PROJECT')
        self.name = "AutoML_Regressor"
        
        # Initialize prediction client
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if credentials_path and os.path.exists(credentials_path):
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            self.prediction_client = automl.PredictionServiceClient(credentials=credentials)
        else:
            self.prediction_client = automl.PredictionServiceClient()
        
        # Model path
        self.model_path = f"projects/{self.project_id}/locations/us-central1/models/{self.model_id}"
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict option prices using AutoML
        
        Args:
            features: DataFrame with input features
            
        Returns:
            Array of predicted prices
        """
        predictions = []
        
        # Convert DataFrame to AutoML format
        for _, row in features.iterrows():
            # Create payload
            payload = {
                "row": {
                    "values": [str(val) for val in row.values]
                }
            }
            
            # Make prediction request
            try:
                response = self.prediction_client.predict(
                    name=self.model_path,
                    payload=payload
                )
                
                # Extract prediction
                if response.payload:
                    pred_value = float(response.payload[0].tables.value)
                    predictions.append(pred_value)
                else:
                    predictions.append(0.0)
            except Exception as e:
                print(f"AutoML prediction error: {e}")
                predictions.append(0.0)
        
        return np.array(predictions)
    
    def predict_batch(self, features: pd.DataFrame, batch_size: int = 100) -> np.ndarray:
        """
        Predict option prices in batches
        
        Args:
            features: DataFrame with input features
            batch_size: Batch size for predictions
            
        Returns:
            Array of predicted prices
        """
        all_predictions = []
        
        for i in range(0, len(features), batch_size):
            batch = features.iloc[i:i+batch_size]
            batch_predictions = self.predict(batch)
            all_predictions.extend(batch_predictions)
        
        return np.array(all_predictions)
