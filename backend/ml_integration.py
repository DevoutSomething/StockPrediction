from tf_train import (StockTradingEnvironment, PPOAgent, DQNAgent, 
                       StockDataPreprocessor, train_rl_agent, evaluate_agent)
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional
from pathlib import Path
import pickle

# Create model directory if it doesn't exist
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

class MLModelManager:
    """
    Manages ML models for stock prediction and trading
    """
    def __init__(self, cache_dir: str = "models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.models = {}  # Cache for loaded models
        self.preprocessors = {}  # Cache for preprocessors
        
    def get_model_path(self, symbol: str, algorithm: str) -> Dict[str, Path]:
        """Get paths for model files"""
        if algorithm == "DQN":
            model_path = self.cache_dir / f"{symbol}_{algorithm}_best.h5"
            return {"model": model_path}
        else:  # PPO
            actor_path = self.cache_dir / f"{symbol}_{algorithm}_actor_best.h5"
            critic_path = self.cache_dir / f"{symbol}_{algorithm}_critic_best.h5"
            return {"actor": actor_path, "critic": critic_path}
    
    def get_preprocessor_path(self, symbol: str) -> Path:
        """Get path for preprocessor file"""
        return self.cache_dir / f"{symbol}_preprocessor.pkl"
    
    def has_trained_model(self, symbol: str, algorithm: str) -> bool:
        """Check if a trained model exists for the given symbol and algorithm"""
        paths = self.get_model_path(symbol, algorithm)
        if algorithm == "DQN":
            return paths["model"].exists()
        else:  # PPO
            return paths["actor"].exists() and paths["critic"].exists()
    
    def has_preprocessor(self, symbol: str) -> bool:
        """Check if a preprocessor exists for the given symbol"""
        return self.get_preprocessor_path(symbol).exists()
    
    def save_preprocessor(self, symbol: str, preprocessor: StockDataPreprocessor):
        """Save preprocessor to file"""
        path = self.get_preprocessor_path(symbol)
        with open(path, 'wb') as f:
            pickle.dump(preprocessor, f)
    
    def load_preprocessor(self, symbol: str) -> Optional[StockDataPreprocessor]:
        """Load preprocessor from file"""
        if symbol in self.preprocessors:
            return self.preprocessors[symbol]
            
        path = self.get_preprocessor_path(symbol)
        if not path.exists():
            return None
            
        try:
            with open(path, 'rb') as f:
                preprocessor = pickle.load(f)
                self.preprocessors[symbol] = preprocessor
                return preprocessor
        except Exception as e:
            print(f"Error loading preprocessor for {symbol}: {e}")
            return None
    
    def get_or_train_model(self, symbol: str, algorithm: str = "PPO", force_retrain: bool = False) -> Dict:
        """
        Get a trained model for the symbol, training it if necessary
        
        Args:
            symbol: Stock symbol
            algorithm: 'DQN' or 'PPO'
            force_retrain: Force retraining even if model exists
            
        Returns:
            Dictionary with agent, preprocessor and training info
        """
        # Generate a cache key
        cache_key = f"{symbol}_{algorithm}"
        
        # Check if we already have this model loaded
        if not force_retrain and cache_key in self.models:
            return self.models[cache_key]
            
        # Check if we have a saved model
        if not force_retrain and self.has_trained_model(symbol, algorithm) and self.has_preprocessor(symbol):
            # Load the model
            if algorithm == "DQN":
                # Create preprocessor
                preprocessor = self.load_preprocessor(symbol)
                if preprocessor is None:
                    print(f"Preprocessor not found for {symbol}, training new model")
                    return self.train_new_model(symbol, algorithm)
                
                # Get sample data to determine state dimensions
                normalized_data, _ = preprocessor.prepare_data(symbol, period="1mo")
                env = StockTradingEnvironment(normalized_data)
                
                # Create and load agent
                agent = DQNAgent(state_dim=env.feature_dim, action_dim=env.action_space)
                model_path = self.get_model_path(symbol, algorithm)["model"]
                agent.load(str(model_path))
                
            else:  # PPO
                # Create preprocessor
                preprocessor = self.load_preprocessor(symbol)
                if preprocessor is None:
                    print(f"Preprocessor not found for {symbol}, training new model")
                    return self.train_new_model(symbol, algorithm)
                
                # Get sample data to determine state dimensions
                normalized_data, _ = preprocessor.prepare_data(symbol, period="1mo")
                env = StockTradingEnvironment(normalized_data)
                
                # Create and load agent
                agent = PPOAgent(state_dim=env.feature_dim, action_dim=env.action_space)
                paths = self.get_model_path(symbol, algorithm)
                agent.load(str(paths["actor"]), str(paths["critic"]))
            
            # Cache the loaded model
            result = {
                "agent": agent,
                "preprocessor": preprocessor,
                "newly_trained": False
            }
            self.models[cache_key] = result
            return result
        
        # Train a new model
        return self.train_new_model(symbol, algorithm)
    
    def train_new_model(self, symbol: str, algorithm: str) -> Dict:
        """Train a new model for the given symbol"""
        print(f"Training new {algorithm} model for {symbol}...")
        start_time = time.time()
        
        # Train the model
        agent, preprocessor, rewards, portfolio_values = train_rl_agent(
            symbol=symbol,
            algorithm=algorithm,
            episodes=50  # Adjust based on your needs
        )
        
        # Cache the preprocessor
        self.save_preprocessor(symbol, preprocessor)
        
        # Cache the model
        cache_key = f"{symbol}_{algorithm}"
        result = {
            "agent": agent,
            "preprocessor": preprocessor,
            "newly_trained": True,
            "training_time": time.time() - start_time,
            "final_portfolio_value": portfolio_values[-1] if portfolio_values else None
        }
        self.models[cache_key] = result
        
        return result
    
    def predict_trades(self, symbol: str, days: int = 30, algorithm: str = "PPO") -> Dict:
        """
        Predict trades for the next N days
        
        Args:
            symbol: Stock symbol
            days: Number of days to predict
            algorithm: 'DQN' or 'PPO'
            
        Returns:
            Dictionary with prediction results
        """
        # Get or train the model
        model_data = self.get_or_train_model(symbol, algorithm)
        agent = model_data["agent"]
        preprocessor = model_data["preprocessor"]
        
        # Get recent data for prediction
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        normalized_data, original_data = preprocessor.prepare_data(
            symbol, 
            period="2mo",  # Get 2 months of data for context
            is_training=False
        )
        
        # Create environment for prediction
        env = StockTradingEnvironment(normalized_data)
        state = env.reset()
        
        # Run the agent to get predictions
        actions = []
        portfolio_values = [env.portfolio_value]
        states = []
        
        # Store the initial state
        states.append({
            'step': 0,
            'price': original_data['Close'].iloc[env.current_step],
            'portfolio_value': env.portfolio_value
        })
        
        # Simulate trading
        done = False
        while not done:
            # Get action from agent
            if algorithm == "DQN":
                action = agent.act(state, training=False)
            else:  # PPO
                action, _, _ = agent.act(state)
            
            # Take the action in the environment
            next_state, reward, done, info = env.step(action)
            state = next_state
            
            # Record the action and portfolio value
            actions.append(action)
            portfolio_values.append(env.portfolio_value)
            
            # Record this state
            if env.current_step < len(original_data):
                states.append({
                    'step': env.current_step,
                    'price': original_data['Close'].iloc[env.current_step],
                    'portfolio_value': env.portfolio_value,
                    'action': 'buy' if action == 2 else 'hold' if action == 1 else 'sell'
                })
        
        # Calculate performance metrics
        initial_price = original_data['Close'].iloc[0]
        final_price = original_data['Close'].iloc[-1]
        price_change_pct = (final_price / initial_price - 1) * 100
        
        agent_return_pct = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
        
        # Extract trading signals
        buy_signals = [i for i, a in enumerate(actions) if a == 2]
        sell_signals = [i for i, a in enumerate(actions) if a == 0]
        
        return {
            'symbol': symbol,
            'algorithm': algorithm,
            'initial_investment': env.initial_balance,
            'final_portfolio_value': portfolio_values[-1],
            'return_pct': agent_return_pct,
            'price_change_pct': price_change_pct,
            'outperformance': agent_return_pct - price_change_pct,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'actions': actions,
            'portfolio_values': portfolio_values,
            'states': states,
            'prediction_date': datetime.now().strftime('%Y-%m-%d')
        }

# Create a global model manager
model_manager = MLModelManager()

def get_trading_recommendation(symbol: str, algorithm: str = "PPO", investment_amount: float = 10000.0) -> Dict:
    """
    Get a trading recommendation for a stock
    
    Args:
        symbol: Stock symbol
        algorithm: RL algorithm to use
        investment_amount: Amount to invest
        
    Returns:
        Trading recommendation
    """
    prediction = model_manager.predict_trades(symbol, algorithm=algorithm)
    
    # Calculate units to buy based on current price
    current_price = prediction['states'][0]['price']
    units = investment_amount / current_price
    
    # Determine if it's a buy/hold/sell recommendation
    last_actions = prediction['actions'][-5:]  # Look at last 5 actions
    buy_count = sum(1 for a in last_actions if a == 2)
    sell_count = sum(1 for a in last_actions if a == 0)
    
    if buy_count > sell_count:
        recommendation = "BUY"
        confidence = buy_count / len(last_actions)
    elif sell_count > buy_count:
        recommendation = "SELL"
        confidence = sell_count / len(last_actions)
    else:
        recommendation = "HOLD"
        confidence = sum(1 for a in last_actions if a == 1) / len(last_actions)
    
    return {
        'symbol': symbol,
        'current_price': current_price,
        'recommendation': recommendation,
        'confidence': confidence,
        'expected_return_pct': prediction['return_pct'],
        'potential_value': investment_amount * (1 + prediction['return_pct']/100),
        'algorithm': algorithm,
        'units_to_trade': units if recommendation == "BUY" else 0,
        'prediction_date': prediction['prediction_date']
    }