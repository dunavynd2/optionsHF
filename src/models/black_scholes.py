
"""
Black-Scholes Model Implementation
Baseline model for European options pricing
"""

import numpy as np
from scipy.stats import norm
from typing import Union


class BlackScholesModel:
    """Black-Scholes model for European options pricing"""
    
    def __init__(self):
        """Initialize Black-Scholes model"""
        self.name = "Black-Scholes"
    
    def calculate_d1(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """
        Calculate d1 parameter
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield
            
        Returns:
            d1 value
        """
        return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    def calculate_d2(self, d1: float, sigma: float, T: float) -> float:
        """
        Calculate d2 parameter
        
        Args:
            d1: d1 value
            sigma: Volatility
            T: Time to expiration (years)
            
        Returns:
            d2 value
        """
        return d1 - sigma * np.sqrt(T)
    
    def price_call(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """
        Price European call option
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield
            
        Returns:
            Call option price
        """
        if T <= 0:
            return max(S - K, 0)
        
        d1 = self.calculate_d1(S, K, T, r, sigma, q)
        d2 = self.calculate_d2(d1, sigma, T)
        
        call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return max(call_price, 0)
    
    def price_put(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """
        Price European put option
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield
            
        Returns:
            Put option price
        """
        if T <= 0:
            return max(K - S, 0)
        
        d1 = self.calculate_d1(S, K, T, r, sigma, q)
        d2 = self.calculate_d2(d1, sigma, T)
        
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        return max(put_price, 0)
    
    def predict(self, features: dict) -> float:
        """
        Predict option price using Black-Scholes
        
        Args:
            features: Dictionary with keys:
                - underlying_price: Current stock price
                - strike_price: Strike price
                - time_to_expiration: Time to expiration (years)
                - risk_free_rate: Risk-free rate
                - volatility: Implied volatility (estimated from historical data)
                - dividend_yield: Dividend yield
                - is_call: 1 for call, 0 for put
                
        Returns:
            Predicted option price
        """
        S = features['underlying_price']
        K = features['strike_price']
        T = features['time_to_expiration']
        r = features['risk_free_rate']
        sigma = features.get('volatility', 0.3)  # Default volatility if not provided
        q = features.get('dividend_yield', 0.0)
        is_call = features.get('is_call', 1)
        
        if is_call:
            return self.price_call(S, K, T, r, sigma, q)
        else:
            return self.price_put(S, K, T, r, sigma, q)
    
    def predict_batch(self, features_list: list) -> np.ndarray:
        """
        Predict option prices for multiple options
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            Array of predicted prices
        """
        return np.array([self.predict(features) for features in features_list])
