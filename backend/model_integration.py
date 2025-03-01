"""Module to integrate tf_train.py models with FastAPI endpoints"""
from backend.tf_train import (StockDataPreprocessor, train_rl_agent, 
                     evaluate_agent, multi_stock_portfolio_optimization)
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
from .tf_train import StockTradingEnvironment
# Create model cache directory
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)

class ModelManager:
    """Class to manage model training, storage, and retrieval"""
    
    @staticmethod
    def get_model_path(symbol: str, algorithm: str = "DQN") -> Path:
        """Get the path to a saved model"""
        return MODEL_DIR / f"{symbol}_{algorithm}_model.h5"
    
    @staticmethod
    def get_preprocessor_path(symbol: str) -> Path:
        """Get the path to a saved preprocessor"""
        return MODEL_DIR / f"{symbol}_preprocessor.pkl"
    
    @staticmethod
    def get_or_train_model(symbol: str, algorithm: str = "DQN", force_retrain: bool = False) -> Dict[str, Any]:
        """Get a trained model or train a new one if needed"""
        model_path = ModelManager.get_model_path(symbol, algorithm)
        preprocessor_path = ModelManager.get_preprocessor_path(symbol)
        
        # Check if we need to train a new model
        if force_retrain or not model_path.exists() or not preprocessor_path.exists():
            print(f"Training new {algorithm} model for {symbol}...")
            agent, preprocessor, rewards, portfolio_values = train_rl_agent(
                symbol=symbol,
                algorithm=algorithm,
                episodes=20,  # Reduced for faster training in production
                visualize=False
            )
            
            # Save model metadata
            metadata = {
                "symbol": symbol,
                "algorithm": algorithm,
                "training_date": datetime.now().isoformat(),
                "final_reward": rewards[-1] if rewards else 0,
                "final_portfolio_value": portfolio_values[-1] if portfolio_values else 0
            }
            
            with open(MODEL_DIR / f"{symbol}_{algorithm}_metadata.json", "w") as f:
                json.dump(metadata, f)
                
            return {
                "agent": agent,
                "preprocessor": preprocessor,
                "metadata": metadata,
                "newly_trained": True
            }
        else:
            # Load existing model and metadata
            with open(MODEL_DIR / f"{symbol}_{algorithm}_metadata.json", "r") as f:
                metadata = json.load(f)
                
            # You would need to implement loading logic based on your agent's save format
            # For this example, we'll retrain if loading fails
            try:
                if algorithm == "DQN":
                    from tf_train import DQNAgent
                    agent = DQNAgent(state_dim=20, action_dim=3)  # Placeholder dimensions
                    agent.load(str(model_path))
                else:
                    from tf_train import PPOAgent
                    agent = PPOAgent(state_dim=20, action_dim=3)  # Placeholder dimensions
                    agent.load(str(model_path))
                    
                preprocessor = StockDataPreprocessor()
                # Load preprocessor state if needed
                
                return {
                    "agent": agent,
                    "preprocessor": preprocessor,
                    "metadata": metadata,
                    "newly_trained": False
                }
            except Exception as e:
                print(f"Error loading model: {e}, retraining...")
                return ModelManager.get_or_train_model(symbol, algorithm, force_retrain=True)
    
    @staticmethod
    def get_prediction(symbol: str, days_ahead: int = 7) -> Dict[str, float]:
        """Get stock price prediction for specified days ahead"""
        # Get or train the model
        model_info = ModelManager.get_or_train_model(symbol)
        agent = model_info["agent"]
        preprocessor = model_info["preprocessor"]
        
        # Get current stock data
        normalized_data, original_data = preprocessor.prepare_data(symbol, period="6mo")
        
        # Use the last available price as current price
        current_price = original_data["Close"].iloc[-1]
        
        # For a simple prediction, we can use the agent to simulate trading for N days
        # and calculate the expected return
        env = StockTradingEnvironment(normalized_data[-100:])  # Use last 100 days
        state = env.reset()
        
        # Simulate trading for days_ahead steps
        for _ in range(days_ahead):
            action = agent.predict(state)
            state, _, done, _ = env.step(action)
            if done:
                break
        
        # Calculate final portfolio value
        final_portfolio_value = env.portfolio_value
        initial_portfolio_value = env.initial_balance
        
        # Calculate predicted price
        predicted_return = (final_portfolio_value / initial_portfolio_value - 1) * 100
        predicted_price = current_price * (1 + predicted_return / 100)
        
        return {
            "current_price": current_price,
            "predicted_price": predicted_price,
            "predicted_return": predicted_return,
            "days_ahead": days_ahead,
            "confidence": 0.75  # Placeholder - implement confidence calculation
        }
    
    @staticmethod
    def optimize_portfolio(symbols: List[str], investment_amount: float, risk_factor: float = 0.5) -> Dict[str, Any]:
        """Optimize a portfolio of stocks"""
        portfolio_summary, agents, preprocessors = multi_stock_portfolio_optimization(
            symbols=symbols,
            investment_amount=investment_amount,
            algorithm="DQN",
            risk_factor=risk_factor,
            rebalance_frequency=30,
            visualize=False
        )
        
        return portfolio_summary 