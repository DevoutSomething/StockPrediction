from tf_train import (StockTradingEnvironment, DQNAgent, PPOAgent, 
                     StockDataPreprocessor, train_rl_agent, evaluate_agent,multi_stock_portfolio_optimization)
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
os.environ["DB_URL"] = "mysql+mysqlconnector://root:NewStrongPassword@localhost:3306/stock_prediction_test"

def test_single_stock_trading():
    """Test the trading system with a single stock"""
    
    # Test parameters
    symbol = "AAPL"  # Using Apple as a test stock
    test_episodes = 3  # Reduced episodes for testing
    print(f"\nTesting trading system with {symbol}")
    print("-" * 50)
    
    try:
        # Test data preprocessing
        print("\n1. Testing data preprocessing...")
        preprocessor = StockDataPreprocessor()
        normalized_data, original_data = preprocessor.prepare_data(symbol, period="1y")
        print(f"✓ Successfully preprocessed data: {len(normalized_data)} data points")
        print(f"✓ Features included: {', '.join(normalized_data.columns)}")
        
        # Test environment creation
        print("\n2. Testing environment creation...")
        env = StockTradingEnvironment(normalized_data)
        state = env.reset()
        print(f"✓ Environment created with state dimension: {len(state)}")
        
        # Test both DQN and PPO agents
        for algorithm in ["DQN", "PPO"]:
            print(f"\n3. Testing {algorithm} agent...")
            
            # Train agent
            print(f"Training {algorithm} agent for {test_episodes} episodes...")
            agent, preprocessor, rewards, portfolio_values = train_rl_agent(
                symbol=symbol,
                algorithm=algorithm,
                episodes=test_episodes
            )
            print(f"✓ Successfully trained {algorithm} agent")
            print(f"✓ Final portfolio value: ${portfolio_values[-1]:.2f}")
            
            # Test evaluation
            print(f"\n4. Testing {algorithm} agent evaluation...")
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            eval_results = evaluate_agent(
                agent=agent,
                symbol=symbol,
                algorithm=algorithm,
                preprocessor=preprocessor,
                start_date=start_date
            )
            
            if eval_results:
                print(f"✓ Evaluation completed successfully")
                print(f"✓ Agent return: {eval_results['agent_return']:.2f}%")
                print(f"✓ Sharpe ratio: {eval_results['sharpe_ratio']:.4f}")
            
    except Exception as e:
        print(f"\n✗ Test failed: {str(e)}")
        raise

def test_portfolio_optimization():
    """Test portfolio optimization with multiple stocks"""
    
    # Test parameters
    symbols = ["AAPL", "MSFT", "GOOGL"]  # Test with major tech stocks
    investment_amount = 10000.0
    risk_factor = 0.5
    
    print("\nTesting portfolio optimization")
    print("-" * 50)
    
    try:
        print(f"\nOptimizing portfolio for: {', '.join(symbols)}")
        portfolio_summary, agents, preprocessors = multi_stock_portfolio_optimization(
            symbols=symbols,
            investment_amount=investment_amount,
            algorithm="PPO",  # PPO typically works better for portfolio optimization
            risk_factor=risk_factor,
            rebalance_frequency=30
        )
        
        print("\n✓ Portfolio optimization completed successfully")
        print(f"✓ Expected portfolio return: {portfolio_summary['expected_return']:.2f}%")
        print(f"✓ Expected Sharpe ratio: {portfolio_summary['expected_sharpe']:.4f}")
        
        # Print allocations
        print("\nPortfolio allocations:")
        for symbol, allocation in portfolio_summary['allocations'].items():
            print(f"✓ {symbol}: {allocation*100:.2f}%")
            
    except Exception as e:
        print(f"\n✗ Portfolio optimization test failed: {str(e)}")
        raise

def test_stock_data_preprocessor():
    """Test the StockDataPreprocessor class"""
    preprocessor = StockDataPreprocessor()
    
    # Test with mock data
    with patch('yfinance.download') as mock_download:
        # Create mock data
        dates = pd.date_range(end=pd.Timestamp.now(), periods=100)
        mock_data = pd.DataFrame({
            'Open': np.random.randn(100) * 10 + 150,
            'High': np.random.randn(100) * 10 + 155,
            'Low': np.random.randn(100) * 10 + 145,
            'Close': np.random.randn(100) * 10 + 150,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        mock_download.return_value = mock_data
        
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
        
        # Test normalization
        normalized = preprocessor.normalize_data(indicators)
        assert normalized.max().max() <= 1.0
        assert normalized.min().min() >= 0.0

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