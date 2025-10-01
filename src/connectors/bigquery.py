
"""
BigQuery Data Connector
Fetches options data from BigQuery tables based on the Entity Relationship Diagram
"""

import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account


class BigQueryConnector:
    """Connector for fetching options data from BigQuery"""
    
    def __init__(self, project_id: Optional[str] = None, dataset_id: Optional[str] = None):
        """
        Initialize BigQuery connector
        
        Args:
            project_id: Google Cloud project ID
            dataset_id: BigQuery dataset ID
        """
        self.project_id = project_id or os.getenv('GOOGLE_CLOUD_PROJECT')
        self.dataset_id = dataset_id or os.getenv('BIGQUERY_DATASET', 'options_data')
        
        # Initialize client
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if credentials_path and os.path.exists(credentials_path):
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            self.client = bigquery.Client(credentials=credentials, project=self.project_id)
        else:
            self.client = bigquery.Client(project=self.project_id)
    
    def get_security_prices(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch security prices for a given symbol and date range
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with security prices
        """
        query = f"""
        SELECT 
            sp.date,
            sp.security_id,
            sp.open_price,
            sp.high_price,
            sp.low_price,
            sp.close_price,
            sp.volume,
            snh.symbol,
            snh.security_name
        FROM `{self.project_id}.{self.dataset_id}.Security_Prices` sp
        JOIN `{self.project_id}.{self.dataset_id}.Security_Name_History` snh
            ON sp.security_id = snh.security_id
        WHERE snh.symbol = @symbol
            AND sp.date BETWEEN @start_date AND @end_date
        ORDER BY sp.date DESC
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("symbol", "STRING", symbol),
                bigquery.ScalarQueryParameter("start_date", "DATE", start_date.date()),
                bigquery.ScalarQueryParameter("end_date", "DATE", end_date.date()),
            ]
        )
        
        return self.client.query(query, job_config=job_config).to_dataframe()
    
    def get_options_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch options data with all required features for model prediction
        
        Args:
            symbol: Underlying stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with options data and features
        """
        query = f"""
        SELECT 
            op.date,
            op.option_id,
            oi.underlying_security_id,
            oi.strike_price,
            oi.expiration_date,
            oi.option_type,
            op.bid_price,
            op.ask_price,
            op.last_price,
            op.volume,
            op.open_interest,
            sp.close_price as underlying_price,
            snh.symbol as underlying_symbol,
            DATE_DIFF(oi.expiration_date, op.date, DAY) as days_to_expiration,
            zcyc.yield_rate as risk_free_rate,
            idy.dividend_yield
        FROM `{self.project_id}.{self.dataset_id}.Options_Prices` op
        JOIN `{self.project_id}.{self.dataset_id}.Options_Info` oi
            ON op.option_id = oi.option_id
        JOIN `{self.project_id}.{self.dataset_id}.Security_Prices` sp
            ON oi.underlying_security_id = sp.security_id
            AND op.date = sp.date
        JOIN `{self.project_id}.{self.dataset_id}.Security_Name_History` snh
            ON oi.underlying_security_id = snh.security_id
        LEFT JOIN `{self.project_id}.{self.dataset_id}.Zero_Coupon_Yield_Curve` zcyc
            ON op.date = zcyc.date
            AND CAST(DATE_DIFF(oi.expiration_date, op.date, DAY) / 365.25 AS INT64) = zcyc.maturity_years
        LEFT JOIN `{self.project_id}.{self.dataset_id}.Index_Dividend_Yield` idy
            ON op.date = idy.date
        WHERE snh.symbol = @symbol
            AND op.date BETWEEN @start_date AND @end_date
            AND oi.expiration_date > op.date
        ORDER BY op.date DESC, oi.expiration_date, oi.strike_price
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("symbol", "STRING", symbol),
                bigquery.ScalarQueryParameter("start_date", "DATE", start_date.date()),
                bigquery.ScalarQueryParameter("end_date", "DATE", end_date.date()),
            ]
        )
        
        return self.client.query(query, job_config=job_config).to_dataframe()
    
    def get_dividend_history(self, symbol: str, lookback_days: int = 365) -> pd.DataFrame:
        """
        Fetch dividend distribution history
        
        Args:
            symbol: Stock symbol
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame with dividend history
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        query = f"""
        SELECT 
            ddh.ex_date,
            ddh.security_id,
            ddh.dividend_amount,
            ddh.dividend_type,
            snh.symbol
        FROM `{self.project_id}.{self.dataset_id}.Dividend_Distribution_History` ddh
        JOIN `{self.project_id}.{self.dataset_id}.Security_Name_History` snh
            ON ddh.security_id = snh.security_id
        WHERE snh.symbol = @symbol
            AND ddh.ex_date BETWEEN @start_date AND @end_date
        ORDER BY ddh.ex_date DESC
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("symbol", "STRING", symbol),
                bigquery.ScalarQueryParameter("start_date", "DATE", start_date.date()),
                bigquery.ScalarQueryParameter("end_date", "DATE", end_date.date()),
            ]
        )
        
        return self.client.query(query, job_config=job_config).to_dataframe()
    
    def prepare_features(self, options_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for model prediction based on the research paper
        
        Args:
            options_df: Raw options data from BigQuery
            
        Returns:
            DataFrame with engineered features
        """
        df = options_df.copy()
        
        # Calculate moneyness
        df['moneyness'] = df['underlying_price'] / df['strike_price']
        
        # Calculate time to expiration in years
        df['time_to_expiration'] = df['days_to_expiration'] / 365.25
        
        # Calculate mid price
        df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2
        
        # Fill missing risk-free rate with default (e.g., 0.05)
        df['risk_free_rate'] = df['risk_free_rate'].fillna(0.05)
        
        # Fill missing dividend yield with 0
        df['dividend_yield'] = df['dividend_yield'].fillna(0.0)
        
        # Create option type indicator (1 for call, 0 for put)
        df['is_call'] = (df['option_type'] == 'CALL').astype(int)
        
        # Calculate implied volatility proxy using bid-ask spread
        df['bid_ask_spread'] = df['ask_price'] - df['bid_price']
        df['relative_spread'] = df['bid_ask_spread'] / df['mid_price']
        
        return df
