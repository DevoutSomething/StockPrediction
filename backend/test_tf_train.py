import pytest
from unittest.mock import patch, MagicMock
import os
import numpy as np
import pandas as pd
from tf_train import (StockTradingEnvironment, DQNAgent, PPOAgent, 
                     StockDataPreprocessor, train_rl_agent, 
                     evaluate_agent, multi_stock_portfolio_optimization)

# Set environment variables for testing
os.environ["TESTING"] = "1"
os.environ["USE_MOCK_DATA"] = "1"  # Always use mock data for testing

# Create a pytest fixture for common test objects
@pytest.fixture
def mock_stock_data():
    """Create mock stock data for testing"""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100)
    data = pd.DataFrame({
        'Open': np.random.randn(100) * 10 + 150,
        'High': np.random.randn(100) * 10 + 155,
        'Low': np.random.randn(100) * 10 + 145,
        'Close': np.random.randn(100) * 10 + 150,
        'Volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    return data

def test_single_stock_trading():
    """Test the trading system with a single stock"""
    
    # Ensure we're using mock data for testing
    os.environ["USE_MOCK_DATA"] = "1"
    
    # Test parameters
    symbol = "AAPL"
    test_episodes = 2  # Use a small number for faster tests
    
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
    
    # Test agent training with actual function
    for algorithm in ["DQN"]:
        agent, prep, rewards, portfolio_values = train_rl_agent(
            symbol=symbol,
            algorithm=algorithm,
            episodes=test_episodes
        )
        
        # Instead of checking exact values, verify the structure and types
        assert isinstance(rewards, list)
        assert len(rewards) == test_episodes
        assert all(isinstance(r, float) for r in rewards)
        
        assert isinstance(portfolio_values, list)
        assert len(portfolio_values) == test_episodes
        assert all(isinstance(v, float) for v in portfolio_values)
        
        # Check that the agent is properly initialized
        assert agent is not None
        
    print("✓ Single stock trading test passed")

def test_portfolio_optimization(monkeypatch):
    """Test portfolio optimization with multiple stocks"""
    
    # Create mock return values
    mock_agent = MagicMock()
    mock_preprocessor = StockDataPreprocessor()
    mock_rewards = [1, 2, 3]
    mock_portfolio_values = [10000, 10500, 11000]
    
    # Define our mock functions
    def mock_train_function(symbol, algorithm, episodes):
        return (mock_agent, mock_preprocessor, mock_rewards, mock_portfolio_values)
    
    def mock_evaluate_function(agent, symbol, algorithm, preprocessor, start_date, end_date=None):
        return {
            'portfolio_value': 12000.0,
            'agent_return': 20.0,
            'buy_hold_return': 15.0,
            'sharpe_ratio': 1.5,
            'actions': [1, 2, 0, 1, 1]
        }
    
    # Apply the monkeypatches
    monkeypatch.setattr("tf_train.train_rl_agent", mock_train_function)
    monkeypatch.setattr("tf_train.evaluate_agent", mock_evaluate_function)
    
    # Test parameters
    symbols = ["AAPL", "MSFT", "GOOGL"]
    investment_amount = 10000.0
    risk_factor = 0.69
    
    print("\nTesting portfolio optimization")
    print("-" * 50)
    
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
    
    print("\n✓ Portfolio optimization completed successfully")

def test_stock_data_preprocessor(mock_stock_data):
    """Test the StockDataPreprocessor class using our fixture"""
    preprocessor = StockDataPreprocessor()
    
    # Test with patch for yfinance.download
    with patch('yfinance.download', return_value=mock_stock_data):
        # Test prepare_data
        normalized_data, original_data = preprocessor.prepare_data("AAPL", period="1y")
        
        # Verify the data was processed correctly
        assert not normalized_data.empty
        assert not original_data.empty
        
        # Test technical indicators
        indicators = preprocessor.add_technical_indicators(mock_stock_data)
        assert 'MA5' in indicators.columns
        assert 'RSI' in indicators.columns
        
        # Test normalization
        normalized = preprocessor.normalize_data(indicators)
        
        # Allow slight tolerance for floating point comparison
        assert np.all(normalized.max() <= 1.01)
        assert np.all(normalized.min() >= -0.01)
        
        print("✓ Stock data preprocessor test passed")

def test_trading_environment(mock_stock_data):
    """Test the StockTradingEnvironment class using our fixture"""
    # Add technical indicators to the mock data
    data = mock_stock_data.copy()
    data['MA5'] = data['Close'].rolling(window=5).mean().fillna(0)
    data['RSI'] = np.random.rand(len(data)) * 100
    
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
    
    print("✓ Trading environment test passed")