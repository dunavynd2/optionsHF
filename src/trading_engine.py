
"""
Alpaca Trading Engine
Handles trade execution and backtesting
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import alpaca_trade_api as tradeapi

from src.connectors.bigquery import BigQueryConnector
from src.ensemble import ModelEnsemble


class TradingEngine:
    """Trading engine for options using Alpaca"""
    
    def __init__(self):
        """Initialize trading engine"""
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials not found in environment")
        
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            self.api_key,
            self.secret_key,
            self.base_url,
            api_version='v2'
        )
        
        # Initialize components
        self.bq_connector = BigQueryConnector()
        self.ensemble = ModelEnsemble(use_automl=False)
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        account = self.api.get_account()
        return {
            'equity': float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value)
        }
    
    def analyze_and_trade(self, symbol: str, quantity: int = 1, 
                         confidence_threshold: float = 0.7) -> Dict:
        """
        Analyze option and execute trade if confidence is high
        
        Args:
            symbol: Stock symbol
            quantity: Number of contracts
            confidence_threshold: Minimum confidence to trade
            
        Returns:
            Trade result
        """
        # Fetch recent options data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        options_df = self.bq_connector.get_options_data(symbol, start_date, end_date)
        
        if options_df.empty:
            return {'error': f'No options data for {symbol}'}
        
        # Prepare features
        features_df = self.bq_connector.prepare_features(options_df)
        
        # Get ensemble prediction
        ensemble_pred, metadata = self.ensemble.predict_ensemble(features_df)
        
        # Check confidence
        confidence = metadata['confidence_scores'][0]
        
        result = {
            'symbol': symbol,
            'prediction': float(ensemble_pred[0]),
            'confidence': float(confidence),
            'timestamp': datetime.now().isoformat()
        }
        
        if confidence >= confidence_threshold:
            # Execute trade (placeholder - Alpaca doesn't support options directly)
            result['trade_signal'] = 'BUY'
            result['message'] = f'High confidence ({confidence:.2%}) - Trade signal generated'
        else:
            result['trade_signal'] = 'HOLD'
            result['message'] = f'Low confidence ({confidence:.2%}) - No trade'
        
        return result
    
    def backtest(self, symbol: str, start_date: datetime, end_date: datetime,
                initial_capital: float = 10000.0) -> Dict:
        """
        Backtest trading strategy
        
        Args:
            symbol: Stock symbol
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Initial capital
            
        Returns:
            Backtest results
        """
        # Fetch historical options data
        options_df = self.bq_connector.get_options_data(symbol, start_date, end_date)
        
        if options_df.empty:
            return {'error': f'No historical data for {symbol}'}
        
        # Prepare features
        features_df = self.bq_connector.prepare_features(options_df)
        
        # Get predictions
        predictions, metadata = self.ensemble.predict_ensemble(features_df)
        
        # Calculate returns (simplified)
        actual_prices = features_df['mid_price'].values
        predicted_prices = predictions
        
        # Calculate metrics
        mae = abs(predicted_prices - actual_prices).mean()
        mape = (abs(predicted_prices - actual_prices) / actual_prices * 100).mean()
        
        results = {
            'symbol': symbol,
            'period': f'{start_date.date()} to {end_date.date()}',
            'trades_analyzed': len(features_df),
            'mae': float(mae),
            'mape': float(mape),
            'initial_capital': initial_capital,
            'predictions': predictions.tolist(),
            'actual_prices': actual_prices.tolist(),
            'confidence_scores': metadata['confidence_scores'].tolist()
        }
        
        return results


def main():
    """Main function for testing"""
    engine = TradingEngine()
    
    # Get account info
    account = engine.get_account_info()
    print(f"Account Info: {account}")
    
    # Analyze and trade
    result = engine.analyze_and_trade('AAPL')
    print(f"Trade Result: {result}")


if __name__ == '__main__':
    main()
