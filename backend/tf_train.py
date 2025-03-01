import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
import pandas as pd
from collections import deque
import random
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import datetime
from typing import List, Dict, Tuple, Any, Optional
import os
import json
from pathlib import Path
import time


# Mock data for testing
MOCK_STOCK_DATA = {
    "Open": [150.0, 151.0, 152.0, 153.0, 154.0] * 20,
    "High": [155.0, 156.0, 157.0, 158.0, 159.0] * 20,
    "Low": [149.0, 148.0, 147.0, 146.0, 145.0] * 20,
    "Close": [153.0, 152.0, 154.0, 155.0, 156.0] * 20,
    "Volume": [1000000, 1100000, 1200000, 1300000, 1400000] * 20
}

def get_mock_data(symbol, period="1y"):
    """Return mock stock data for testing"""
    # Create a pandas DataFrame with mock data
    dates = pd.date_range(end=pd.Timestamp.now(), periods=len(MOCK_STOCK_DATA["Open"]))
    mock_df = pd.DataFrame(MOCK_STOCK_DATA, index=dates)
    return mock_df

def get_stock_data(symbol: str, period: str = "1y", start: str = None, end: str = None) -> pd.DataFrame:
    """Helper function to get stock data with rate limit handling"""
    # Add random delay between 1-3 seconds to avoid hitting rate limits
    time.sleep(random.uniform(1, 3))
    
    try:
        # Use either period or start/end dates
        if start and end:
            data = yf.download(symbol, start=start, end=end, progress=False)
        else:
            data = yf.download(symbol, period=period, progress=False)

        if data is None or data.empty:
            print(f"No data received for {symbol}, falling back to mock data")
            return get_mock_data(symbol, period)
            
        return data
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return get_mock_data(symbol, period)

class StockTradingEnvironment:
    """
    A custom environment for stock trading using reinforcement learning
    """
    def __init__(self, 
                 stock_data: pd.DataFrame, 
                 initial_balance: float = 10000.0,
                 max_steps: int = 1000,
                 transaction_fee_percent: float = 0.001,
                 reward_scaling: float = 0.001):
        
        self.stock_data = stock_data
        self.initial_balance = initial_balance
        self.max_steps = max_steps
        self.transaction_fee_percent = transaction_fee_percent
        self.reward_scaling = reward_scaling
        
        # State space dimensions: price data, technical indicators, portfolio info
        self.feature_dim = self.stock_data.shape[1] + 3  # +3 for balance, shares, portfolio value
        
        # Action space: 0 (sell), 1 (hold), 2 (buy)
        self.action_space = 3
        
        # Reset the environment
        self.reset()
    
    def reset(self):
        """Reset the environment for a new episode"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.portfolio_value = self.initial_balance
        self.history = []
        
        # Get the initial state
        return self._get_state()
    
    def _get_state(self):
        """
        Construct the state representation that includes:
        1. Current market data (normalized)
        2. Current portfolio information
        """
        # Get the current market data
        market_data = self.stock_data.iloc[self.current_step].values
        
        # Portfolio information
        portfolio_info = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.shares_held * self.stock_data['Close'].iloc[self.current_step] / self.initial_balance,  # Normalized position value
            self.portfolio_value / self.initial_balance  # Normalized portfolio value
        ])
        
        # Combine market data and portfolio info
        state = np.concatenate([market_data, portfolio_info])
        
        return state
    
    def step(self, action):
        """
        Take an action in the environment:
        - 0: Sell 100% of shares
        - 1: Hold
        - 2: Buy shares with 100% of balance
        """
        # Get current stock price
        current_price = self.stock_data['Close'].iloc[self.current_step]
        
        # Ensure price is not zero to avoid division by zero
        if current_price <= 0:
            current_price = 0.01  # Use a small positive value instead of zero
        
        # Initialize reward
        reward = 0
        done = False
        
        # Execute the trade action
        if action == 0:  # Sell
            if self.shares_held > 0:
                # Calculate the sale amount and transaction fee
                sale_amount = self.shares_held * current_price
                transaction_fee = sale_amount * self.transaction_fee_percent
                
                # Update balance and shares
                self.balance += sale_amount - transaction_fee
                self.shares_held = 0
                
                # Log the action
                self.history.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'price': current_price,
                    'shares': self.shares_held,
                    'balance': self.balance,
                    'portfolio_value': self.portfolio_value
                })
        
        elif action == 2:  # Buy
            if self.balance > 0:
                # Calculate the maximum shares we can buy (with safety check)
                # Ensure we don't get division by zero or very small values
                denominator = current_price * (1 + self.transaction_fee_percent)
                if denominator < 0.01:
                    denominator = 0.01  # Avoid division by very small numbers
                    
                max_shares = self.balance / denominator
                
                # Add safety check to prevent overflow
                if max_shares > 1e9:  # Cap at a billion shares
                    max_shares = 1e9
                    
                # Implement a slightly more complex buying strategy: use 90% of available balance
                shares_to_buy = int(max_shares * 0.9)
                
                # Ensure we buy at least 1 share if possible
                shares_to_buy = max(1, shares_to_buy) if self.balance >= current_price else 0
                
                # Update shares and balance
                cost = shares_to_buy * current_price
                transaction_fee = cost * self.transaction_fee_percent
                self.shares_held += shares_to_buy
                self.balance -= (cost + transaction_fee)
                
                # Log the action
                self.history.append({
                    'step': self.current_step,
                    'action': 'buy',
                    'price': current_price,
                    'shares': shares_to_buy,
                    'balance': self.balance,
                    'portfolio_value': self.portfolio_value
                })
        
        # Hold action doesn't require any portfolio changes
        
        # Calculate portfolio value
        new_portfolio_value = self.balance + (self.shares_held * current_price)
        
        # Calculate reward based on portfolio value change
        reward = ((new_portfolio_value - self.portfolio_value) / self.portfolio_value) * self.reward_scaling
        
        # Update portfolio value
        self.portfolio_value = new_portfolio_value
        
        # Advance to the next step
        self.current_step += 1
        
        # Check if the episode is done
        if self.current_step >= len(self.stock_data) - 1 or self.current_step >= self.max_steps:
            done = True
        
        # Get the next state
        next_state = self._get_state()
        
        return next_state, reward, done, {'portfolio_value': self.portfolio_value}

class DQNAgent:
    """
    Deep Q-Network based agent for stock trading
    """
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 batch_size: int = 64,
                 memory_size: int = 10000):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Initialize replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Build the main model and target model
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self):
        """Build a neural network model for deep Q-learning"""
        model = Sequential()
        model.add(Input(shape=(self.state_dim,)))  # Use Input layer for the input shape
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_dim, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
        
    def update_target_model(self):
        """Update the target model with the weights from the main model"""
        self.target_model.set_weights(self.model.get_weights())
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, training=True):
        """Select an action using epsilon-greedy policy"""
        if training and np.random.rand() <= self.epsilon:
            # Exploration: choose a random action
            return random.randrange(self.action_dim)
        
        # Exploitation: choose the best action according to the model
        act_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size=None):
        """Train the agent with experiences from memory"""
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.memory) < batch_size:
            return
            
        # Sample a batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])
        
        # Predict Q-values for current states
        targets = self.model.predict(states, verbose=0)
        
        # Predict Q-values for next states using target model
        target_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Update Q-values for actions taken
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(target_q_values[i])
                
        # Train the model
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def load(self, name):
        """Load model from file"""
        self.model.load_weights(name)
        
    def save(self, name):
        """Save model to file"""
        # Ensure filename has the correct extension for Keras 3 compatibility
        if not name.endswith('.weights.h5'):
            # Remove .h5 extension if it exists
            if name.endswith('.h5'):
                name = name[:-3]
            # Add the correct extension
            name = name + '.weights.h5'
        self.model.save_weights(name)

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent for stock trading
    Better for risk-adjusted returns and safer trading strategies
    """
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 actor_learning_rate: float = 0.0001,
                 critic_learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 clip_ratio: float = 0.2,
                 batch_size: int = 64):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.batch_size = batch_size
        
        # Initialize memory buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # Build actor (policy) and critic (value) networks
        self.actor = self._build_actor(actor_learning_rate)
        self.critic = self._build_critic(critic_learning_rate)
        
    def _build_actor(self, learning_rate):
        """Build the actor (policy) network"""
        inputs = Input(shape=(self.state_dim,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(self.action_dim, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=learning_rate))
        return model
    
    def _build_critic(self, learning_rate):
        """Build the critic (value) network"""
        inputs = Input(shape=(self.state_dim,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(1, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
        return model
    
    def act(self, state):
        """Select an action based on the current policy"""
        # Get action probabilities from the actor network
        state_input = np.expand_dims(state, axis=0)
        action_probs = self.actor.predict(state_input, verbose=0)[0]
        
        # Sample an action from the probability distribution
        action = np.random.choice(self.action_dim, p=action_probs)
        
        # Get the value estimate from the critic
        value = self.critic.predict(state_input, verbose=0)[0, 0]
        
        # Calculate log probability of the selected action
        log_prob = np.log(action_probs[action] + 1e-10)
        
        return action, value, log_prob
    
    def remember(self, state, action, reward, value, log_prob, done):
        """Store experience in memory"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def _calculate_advantages(self):
        """Calculate advantages and returns for PPO update"""
        advantages = np.zeros(len(self.rewards))
        returns = np.zeros(len(self.rewards))
        
        last_return = 0
        last_advantage = 0
        
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1 or self.dones[t]:
                next_value = 0
            else:
                next_value = self.values[t + 1]
                
            delta = self.rewards[t] + self.gamma * next_value * (1 - self.dones[t]) - self.values[t]
            advantages[t] = delta + self.gamma * 0.95 * last_advantage * (1 - self.dones[t])
            returns[t] = self.rewards[t] + self.gamma * last_return * (1 - self.dones[t])
            
            last_return = returns[t]
            last_advantage = advantages[t]
            
        return advantages, returns
    
    def train(self):
        """Train the agent using PPO algorithm"""
        if len(self.states) < self.batch_size:
            return
            
        # Convert to numpy arrays
        states = np.array(self.states)
        actions = np.array(self.actions)
        old_log_probs = np.array(self.log_probs)
        
        # Calculate advantages and returns
        advantages, returns = self._calculate_advantages()
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)
        
        # PPO update loop
        for _ in range(5):  # Multiple epochs for PPO
            # Sample random mini-batches
            indices = np.random.permutation(len(states))
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Actor (policy) update
                with tf.GradientTape() as tape:
                    # Calculate new log probabilities
                    new_action_probs = self.actor(batch_states)
                    new_log_probs = tf.math.log(tf.reduce_sum(
                        new_action_probs * tf.one_hot(batch_actions, self.action_dim), axis=1) + 1e-10)
                    
                    # Calculate ratio and clipped objective
                    ratio = tf.exp(new_log_probs - batch_old_log_probs)
                    clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    
                    # Calculate surrogate losses
                    surrogate1 = ratio * batch_advantages
                    surrogate2 = clipped_ratio * batch_advantages
                    
                    # Calculate policy loss (negative because we want to maximize the objective)
                    policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
                    
                # Compute policy gradients and apply them
                policy_gradients = tape.gradient(policy_loss, self.actor.trainable_variables)
                self.actor.optimizer.apply_gradients(zip(policy_gradients, self.actor.trainable_variables))
                
                # Critic (value) update
                self.critic.fit(batch_states, batch_returns, verbose=0, epochs=1)
        
        # Clear memory after training
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def save(self, actor_path, critic_path):
        """Save actor and critic models"""
        # Ensure actor filename has the correct extension
        if not actor_path.endswith('.weights.h5'):
            # Remove .h5 extension if it exists
            if actor_path.endswith('.h5'):
                actor_path = actor_path[:-3]
            # Add the correct extension
            actor_path = actor_path + '.weights.h5'
        
        # Ensure critic filename has the correct extension
        if not critic_path.endswith('.weights.h5'):
            # Remove .h5 extension if it exists
            if critic_path.endswith('.h5'):
                critic_path = critic_path[:-3]
            # Add the correct extension
            critic_path = critic_path + '.weights.h5'
        
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        
    def load(self, actor_path, critic_path):
        """Load actor and critic models"""
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)

class StockDataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def prepare_data(self, symbol: str, period: str = "1y", use_mock: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare stock data for training"""
        max_retries = 3
        retry_delay = 5  # Increased delay between retries
        
        for attempt in range(max_retries):
            try:
                # Get data
                if use_mock or os.environ.get("USE_MOCK_DATA") == "1":
                    data = get_mock_data(symbol, period)
                    print(f"Using mock data for {symbol}")
                else:
                    # Add retry mechanism for yfinance with exponential backoff
                    try:
                        # Add delay between requests to avoid rate limiting
                        if attempt > 0:
                            wait_time = retry_delay * (2 ** attempt)
                            print(f"Waiting {wait_time} seconds before retry...")
                            time.sleep(wait_time)
                        
                        data = get_stock_data(symbol, period)
                        
                    except Exception as e:
                        if attempt == max_retries - 1:
                            print(f"All attempts failed, falling back to mock data for {symbol}")
                            data = get_mock_data(symbol, period)
                        else:
                            print(f"Attempt {attempt + 1} failed: {str(e)}")
                            continue

                # Process the data (rest of the method remains the same)
                if data.empty:
                    raise ValueError(f"Empty dataset received for {symbol}")
                
                if len(data) < 2:
                    raise ValueError(f"Insufficient data points for {symbol} (got {len(data)})")
                    
                # Ensure we have the required columns
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_columns = [col for col in required_columns if col not in data.columns]
                if missing_columns:
                    raise ValueError(f"Missing required columns for {symbol}: {missing_columns}")
                
                # Add technical indicators with safety checks
                data = self.add_technical_indicators(data)
                
                # Handle any NaN values
                data = data.ffill().bfill()
                if data.isnull().any().any():
                    print(f"Warning: NaN values found in {symbol} data after forward/backward fill")
                    data = data.dropna()
                
                if len(data) < 30:  # Minimum required length
                    raise ValueError(f"Insufficient data points after preprocessing for {symbol} (got {len(data)}, need at least 30)")
                
                # Normalize the data
                normalized_data = self.normalize_data(data)
                
                return normalized_data, data
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                    time.sleep(retry_delay)
                else:
                    print(f"All attempts failed for {symbol}: {str(e)}")
                    raise ValueError(f"Could not prepare data for {symbol} after {max_retries} attempts: {str(e)}")
            
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators safely"""
        try:
            df = data.copy()
            
            # Ensure we have enough data points
            min_periods = 5
            if len(df) < min_periods:
                raise ValueError("Not enough data points for technical indicators")
                
            # Moving averages
            df['MA5'] = df['Close'].rolling(window=5, min_periods=min_periods).mean()
            df['MA20'] = df['Close'].rolling(window=20, min_periods=min_periods).mean()
            
            # RSI with safety checks
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=min_periods).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=min_periods).mean()
            
            # Avoid division by zero
            rs = gain / loss.replace(0, float('nan'))
            df['RSI'] = 100 - (100 / (1 + rs))
            
            return df
            
        except Exception as e:
            print(f"Error in add_technical_indicators: {str(e)}")
            raise
            
    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize the data safely"""
        try:
            # Select features for normalization
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'RSI']
            
            # Create a copy of the data
            normalized = data[features].copy()
            
            # Handle Volume separately (log transformation)
            normalized['Volume'] = np.log1p(normalized['Volume'])
            
            # Normalize each feature
            for feature in features:
                if feature != 'Volume':  # Volume already transformed
                    feature_data = normalized[feature].values.reshape(-1, 1)
                    normalized[feature] = self.scaler.fit_transform(feature_data)
                    
            return normalized
            
        except Exception as e:
            print(f"Error in normalize_data: {str(e)}")
            raise

def train_rl_agent(
    symbol: str = "AAPL", 
    algorithm: str = "PPO", 
    episodes: int = 50,
    initial_balance: float = 10000.0,  # Add this parameter
    period: str = "5y"  # Add this parameter
):
    """
    Train an RL agent on stock trading
    
    Args:
        symbol: Stock symbol to train on
        algorithm: RL algorithm to use ('DQN' or 'PPO')
        episodes: Number of training episodes
        initial_balance: Starting balance for trading
        period: Data period to train on
    """
    # Preprocess data
    preprocessor = StockDataPreprocessor()
    try:
        normalized_data, original_data = preprocessor.prepare_data(symbol, period=period)
    except Exception as e:
        print(f"Error preparing data: {str(e)}")
        # Fall back to mock data if real data fails
        normalized_data, original_data = preprocessor.prepare_data(symbol, period=period, use_mock=True)
    
    # Create environment with specified initial balance
    env = StockTradingEnvironment(
        normalized_data, 
        initial_balance=initial_balance,
        max_steps=len(normalized_data)-1
    )
    
    # Set up agent based on algorithm
    if algorithm == "DQN":
        agent = DQNAgent(state_dim=env.feature_dim, action_dim=env.action_space)
        target_update_frequency = 10
    elif algorithm == "PPO":
        agent = PPOAgent(state_dim=env.feature_dim, action_dim=env.action_space)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Performance tracking
    best_reward = -np.inf
    rewards_history = []
    portfolio_values = []
    
    # Training loop
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        
        while not done:
            if algorithm == "DQN":
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                
                # Train the agent
                agent.replay()
                
                # Update target network periodically
                if step_count % target_update_frequency == 0:
                    agent.update_target_model()
            
            elif algorithm == "PPO":
                action, value, log_prob = agent.act(state)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, value, log_prob, done)
                state = next_state
                
                # Train the agent after sufficient steps
                if len(agent.states) >= agent.batch_size:
                    agent.train()
            
            total_reward += reward
            step_count += 1
        
        # Additional training for PPO at the end of episode
        if algorithm == "PPO":
            agent.train()
        
        # Track performance
        rewards_history.append(total_reward)
        portfolio_values.append(env.portfolio_value)
        
        # Save best model
        if env.portfolio_value > best_reward:
            best_reward = env.portfolio_value
            if algorithm == "DQN":
                agent.save(f"models/{symbol}_{algorithm}_best.h5")
            else:
                agent.save(f"models/{symbol}_{algorithm}_actor_best.h5", f"models/{symbol}_{algorithm}_critic_best.h5")
        
        # Log progress
        print(f"Episode: {episode+1}/{episodes}, Final Portfolio Value: ${env.portfolio_value:.2f}, "
              f"Reward: {total_reward:.4f}, Best: ${best_reward:.2f}")
    
    # Save final model
    if algorithm == "DQN":
        agent.save(f"models/{symbol}_{algorithm}_final.h5")
    else:
        agent.save(f"models/{symbol}_{algorithm}_actor_final.h5", f"models/{symbol}_{algorithm}_critic_final.h5")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(rewards_history)
    plt.title(f'Rewards Over Time ({algorithm} - {symbol})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(2, 1, 2)
    plt.plot(portfolio_values)
    plt.title(f'Portfolio Value Over Time ({algorithm} - {symbol})')
    plt.xlabel('Episode')
    plt.ylabel('Portfolio Value ($)')
    
    plt.tight_layout()
    plt.savefig(f"{symbol}_{algorithm}_training_results.png")
    
    return agent, preprocessor, rewards_history, portfolio_values

def evaluate_agent(agent, symbol: str, algorithm: str, preprocessor: StockDataPreprocessor, 
                   start_date: str, end_date: str = None):
    """
    Evaluate a trained RL agent on out-of-sample data
    
    Args:
        agent: Trained RL agent
        symbol: Stock symbol to evaluate on
        algorithm: RL algorithm used ('DQN' or 'PPO')
        preprocessor: Preprocessor with fitted scalers
        start_date: Start date for evaluation period
        end_date: End date for evaluation period
    """
    # Get evaluation data
    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    stock_data = get_stock_data(symbol, start=start_date, end=end_date)
    
    # Skip if no data
    if stock_data.empty:
        print(f"No data available for {symbol} from {start_date} to {end_date}")
        return None
    
    # Add indicators and normalize
    stock_data_with_indicators = preprocessor.add_technical_indicators(stock_data)
    normalized_data = preprocessor.normalize_data(stock_data_with_indicators, is_training=False)
    
    # Create evaluation environment
    env = StockTradingEnvironment(normalized_data, initial_balance=10000.0)
    
    # Run evaluation
    state = env.reset()
    done = False
    actions_taken = []
    portfolio_values = [env.portfolio_value]
    
    while not done:
        if algorithm == "DQN":
            action = agent.act(state, training=False)
        else:  # PPO
            action, _, _ = agent.act(state)
        
        next_state, reward, done, info = env.step(action)
        state = next_state
        
        actions_taken.append(action)
        portfolio_values.append(env.portfolio_value)
    
    # Calculate metrics
    buy_hold_return = (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0] - 1) * 100
    agent_return = (env.portfolio_value / 10000 - 1) * 100
    
    print(f"\nEvaluation results for {symbol} using {algorithm}:")
    print(f"Start Date: {start_date}, End Date: {end_date}")
    print(f"Final Portfolio Value: ${env.portfolio_value:.2f}")
    print(f"Agent Return: {agent_return:.2f}%")
    print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"Outperformance: {agent_return - buy_hold_return:.2f}%")
    
    # Calculate Sharpe ratio (assuming risk-free rate of 1%)
    returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
    sharpe_ratio = (np.mean(returns) * 252 - 0.01) / (np.std(returns) * np.sqrt(252))
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(stock_data.index, stock_data['Close'])
    plt.title(f'{symbol} Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    
    plt.subplot(3, 1, 2)
    plt.plot(portfolio_values)
    plt.title(f'Portfolio Value Over Time ({algorithm})')
    plt.xlabel('Trading Day')
    plt.ylabel('Value ($)')
    
    plt.subplot(3, 1, 3)
    action_labels = ['Sell', 'Hold', 'Buy']
    # Convert actions to colors for visualization
    action_colors = ['red' if a == 0 else 'gray' if a == 1 else 'green' for a in actions_taken]
    
    for i in range(len(actions_taken)):
        plt.axvline(x=i, color=action_colors[i], alpha=0.2)
        
    plt.title('Agent Actions')
    plt.xlabel('Trading Day')
    plt.yticks([])
    
    # Add a custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.2, label='Buy'),
        Patch(facecolor='gray', alpha=0.2, label='Hold'),
        Patch(facecolor='red', alpha=0.2, label='Sell')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(f"{symbol}_{algorithm}_evaluation_results.png")
    
    return {
        'portfolio_value': env.portfolio_value,
        'agent_return': agent_return,
        'buy_hold_return': buy_hold_return,
        'sharpe_ratio': sharpe_ratio,
        'actions': actions_taken
    }

def multi_stock_portfolio_optimization(symbols: List[str], 
                                      investment_amount: float = 100000.0,
                                      algorithm: str = "PPO",
                                      risk_factor: float = 0.5,  # 0 = lowest risk, 1 = highest risk
                                      rebalance_frequency: int = 30):  # days between rebalancing
    """
    Optimize a portfolio of multiple stocks using reinforcement learning
    
    Args:
        symbols: List of stock symbols to include in portfolio
        investment_amount: Total investment amount in dollars
        algorithm: RL algorithm to use ('DQN' or 'PPO')
        risk_factor: Risk tolerance factor (0-1)
        rebalance_frequency: How often to rebalance the portfolio (in days)
    
    Returns:
        Optimized portfolio allocation and performance metrics
    """
    # Storage for agents and preprocessors
    agents = {}
    preprocessors = {}
    evaluations = {}
    
    # Train an agent for each stock
    for symbol in symbols:
        print(f"\nTraining agent for {symbol}...")
        agent, preprocessor, rewards, values = train_rl_agent(
            symbol=symbol, 
            algorithm=algorithm, 
            episodes=50,  # Can be adjusted based on complexity
            initial_balance=10000.0,  # Add this parameter
            period="5y"  # Add this parameter
        )
        
        agents[symbol] = agent
        preprocessors[symbol] = preprocessor
        
        # Evaluate each agent
        print(f"\nEvaluating agent for {symbol}...")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
        eval_result = evaluate_agent(
            agent=agent,
            symbol=symbol,
            algorithm=algorithm,
            preprocessor=preprocessor,
            start_date=start_date
        )
        
        evaluations[symbol] = eval_result
    
    # Calculate portfolio allocation based on performance and risk factor
    allocations = {}
    total_score = 0
    
    for symbol, eval_result in evaluations.items():
        if eval_result is None:
            continue
            
        # Calculate a combined score based on returns and risk (Sharpe ratio)
        # Higher risk_factor will prioritize returns over Sharpe ratio
        score = (risk_factor * eval_result['agent_return']) + ((1 - risk_factor) * eval_result['sharpe_ratio'] * 100)
        
        # Ensure score is positive (at least 0)
        score = max(0, score)
        allocations[symbol] = score
        total_score += score
    
    # Normalize allocations to sum to 1
    if total_score > 0:
        for symbol in allocations:
            allocations[symbol] = allocations[symbol] / total_score
    else:
        # If all scores are negative or zero, use equal allocation
        for symbol in allocations:
            allocations[symbol] = 1.0 / len(allocations)
    
    # Calculate dollar amounts for each stock
    dollar_allocations = {}
    for symbol, allocation in allocations.items():
        dollar_allocations[symbol] = investment_amount * allocation
    
    # Generate portfolio summary
    portfolio_summary = {
        'allocations': allocations,
        'dollar_allocations': dollar_allocations,
        'expected_return': sum(evaluations[s]['agent_return'] * allocations[s] for s in allocations if evaluations[s] is not None),
        'expected_sharpe': sum(evaluations[s]['sharpe_ratio'] * allocations[s] for s in allocations if evaluations[s] is not None),
        'rebalance_frequency': rebalance_frequency,
        'investment_amount': investment_amount,
        'risk_factor': risk_factor
    }
    
    # Visualize the portfolio allocation
    plt.figure(figsize=(10, 6))
    plt.pie([dollar_allocations[s] for s in dollar_allocations], 
            labels=[f"{s}: ${dollar_allocations[s]:.2f}" for s in dollar_allocations],
            autopct='%1.1f%%')
    plt.title(f'Portfolio Allocation (Risk Factor: {risk_factor})')
    plt.savefig(f"portfolio_allocation_rf{risk_factor}.png")
    
    # Display a summary table
    print("\nPortfolio Optimization Summary:")
    print("=" * 70)
    print(f"Total Investment: ${investment_amount:.2f}")
    print(f"Risk Factor: {risk_factor} (0=lowest risk, 1=highest risk)")
    print(f"Rebalance Frequency: Every {rebalance_frequency} days")
    print(f"Expected Portfolio Return: {portfolio_summary['expected_return']:.2f}%")
    print(f"Expected Portfolio Sharpe Ratio: {portfolio_summary['expected_sharpe']:.4f}")
    print("\nAllocations:")
    for symbol, allocation in sorted(allocations.items(), key=lambda x: x[1], reverse=True):
        print(f"  {symbol}: {allocation*100:.2f}% (${dollar_allocations[symbol]:.2f})")
    
    return portfolio_summary, agents, preprocessors