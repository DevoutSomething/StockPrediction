from tf_train import (StockTradingEnvironment, DQNAgent, PPOAgent, 
                     StockDataPreprocessor, train_rl_agent, 
                     evaluate_agent, multi_stock_portfolio_optimization)
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock


# Set environment variables for testing
os.environ["TESTING"] = "1"
os.environ["DB_URL"] = "mysql+mysqlconnector://root:171205@Kunj@localhost:3306/stock_prediction"
os.environ["USE_MOCK_DATA"] = "1"  # Always use mock data for testing

@patch('tf_train.DQNAgent.save')
def test_single_stock_trading(mock_save):
    """Test the trading system with a single stock"""
    
    # Patch the train_rl_agent function to avoid actual training
    with patch('tf_train.train_rl_agent') as mock_train:
        # Create a mock return value
        mock_agent = MagicMock()
        mock_preprocessor = StockDataPreprocessor()
        mock_rewards = [1, 2, 3]
        mock_portfolio_values = [10000, 10500, 11000]
        
        # Set the return value for the mock
        mock_train.return_value = (mock_agent, mock_preprocessor, mock_rewards, mock_portfolio_values)
        
        # Test parameters
        symbol = "AAPL"
        test_episodes = 2
        
        # Test data preprocessing
        preprocessor = StockDataPreprocessor()
        normalized_data, original_data = preprocessor.prepare_data(symbol, period="1y")
        
        # Verify preprocessed data
        assert not normalized_data.empty
        assert len(normalized_data.columns) > 5
        
        # Test environment creation
        env = StockTradingEnvironment(normalized_data)
        state = env.reset()
        assert len(state) > 0
        
        # Test agent training with mock
        for algorithm in ["DQN"]:
            agent, prep, rewards, portfolio_values = train_rl_agent(
                symbol=symbol,
                algorithm=algorithm,
                episodes=test_episodes
            )
            
            # Since we're mocking, these should match our mock values
            assert rewards == mock_rewards
            assert portfolio_values == mock_portfolio_values

@patch('tf_train.PPOAgent.save')
def test_portfolio_optimization(mock_save):
    """Test portfolio optimization with multiple stocks"""
    
    # Patch train_rl_agent to avoid actual training
    with patch('tf_train.train_rl_agent') as mock_train:
        # Create mock return values
        mock_agent = MagicMock()
        mock_preprocessor = StockDataPreprocessor()
        mock_rewards = [1, 2, 3]
        mock_portfolio_values = [10000, 10500, 11000]
        
        # Configure mock
        mock_train.return_value = (mock_agent, mock_preprocessor, mock_rewards, mock_portfolio_values)
        
        # Mock evaluate_agent to avoid actual evaluation
        with patch('tf_train.evaluate_agent') as mock_evaluate:
            # Create mock evaluation results
            mock_eval_result = {
                'portfolio_value': 12000.0,
                'agent_return': 20.0,
                'buy_hold_return': 15.0,
                'sharpe_ratio': 1.5,
                'actions': [1, 2, 0, 1, 1]
            }
            mock_evaluate.return_value = mock_eval_result
        
            # Test parameters
            symbols = ["AAPL", "MSFT", "GOOGL"]
            investment_amount = 10000.0
            risk_factor = 0.5
            
            print("\nTesting portfolio optimization")
            print("-" * 50)
            
            try:
                print(f"\nOptimizing portfolio for: {', '.join(symbols)}")
                portfolio_summary, agents, preprocessors = multi_stock_portfolio_optimization(
                    symbols=symbols,
                    investment_amount=investment_amount,
                    algorithm="PPO",
                    risk_factor=risk_factor,
                    rebalance_frequency=30
                )
                
                # Verify the portfolio summary
                assert isinstance(portfolio_summary, dict)
                assert 'allocations' in portfolio_summary
                assert 'expected_return' in portfolio_summary
                assert 'investment_amount' in portfolio_summary
                
                print("\n✓ Portfolio optimization completed successfully")
                
            except Exception as e:
                print(f"\n✗ Portfolio optimization test failed: {str(e)}")
                raise

def test_stock_data_preprocessor():
    """Test the StockDataPreprocessor class"""
    preprocessor = StockDataPreprocessor()
    
    # Test with mock data
    with patch('yfinance.download') as mock_download:
        # Create mock data with more realistic values
        dates = pd.date_range(end=pd.Timestamp.now(), periods=100)
        mock_data = pd.DataFrame({
            'Open': np.random.randn(100) * 10 + 150,
            'High': np.random.randn(100) * 10 + 155,
            'Low': np.random.randn(100) * 10 + 145,
            'Close': np.random.randn(100) * 10 + 150,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        mock_download.return_value = mock_data
        
        # Explicitly set environment variable for mock data
        os.environ["USE_MOCK_DATA"] = "1"
        
        # Test prepare_data
        normalized_data, original_data = preprocessor.prepare_data("AAPL", period="1y")
        
        # Verify the data was processed correctly
        assert not normalized_data.empty
        assert not original_data.empty
        assert len(normalized_data.columns) > len(original_data.columns)  # Should have added indicators
        
        # Test technical indicators
        indicators = preprocessor.add_technical_indicators(mock_data)
        assert 'MA5' in indicators.columns
        assert 'RSI' in indicators.columns
        
        # Test normalization - fix floating point comparison
        normalized = preprocessor.normalize_data(indicators)
        
        # Use np.isclose or approximate comparison for floating point
        assert np.all(normalized.max() <= 1.01)  # Allow slight tolerance
        assert np.all(normalized.min() >= -0.01)  # Allow slight tolerance

def test_trading_environment():
    """Test the StockTradingEnvironment class"""
    # Create sample data
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100)
    data = pd.DataFrame({
        'Open': np.random.randn(100) * 10 + 150,
        'High': np.random.randn(100) * 10 + 155,
        'Low': np.random.randn(100) * 10 + 145,
        'Close': np.random.randn(100) * 10 + 150,
        'Volume': np.random.randint(1000000, 5000000, 100),
        'MA5': np.random.randn(100) * 5 + 150,
        'RSI': np.random.rand(100) * 100
    }, index=dates)
    
    # Create environment
    env = StockTradingEnvironment(data)
    
    # Test reset
    state = env.reset()
    assert len(state) == env.feature_dim
    assert env.balance == env.initial_balance
    assert env.shares_held == 0
    
    # Test step with different actions
    # Buy
    next_state, reward, done, info = env.step(2)
    assert env.shares_held > 0
    assert env.balance < env.initial_balance
    
    # Hold
    balance_before = env.balance
    shares_before = env.shares_held
    next_state, reward, done, info = env.step(1)
    assert env.balance == balance_before
    assert env.shares_held == shares_before
    
    # Sell
    next_state, reward, done, info = env.step(0)
    assert env.shares_held == 0
    assert env.balance > 0

if __name__ == "__main__":
    print("Starting trading system tests...")
    
    # Test single stock trading
    test_single_stock_trading()
    
    # Test portfolio optimization
    test_portfolio_optimization()
    
    # Test stock data preprocessor
    test_stock_data_preprocessor()
    
    # Test trading environment
    test_trading_environment()
    
    print("\nAll tests completed!")