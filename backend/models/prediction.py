import tensorflow as tf
import numpy as np
import pandas as pd
import yfinance as yf
import os
import glob
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from collections import deque
import datetime
from sklearn.preprocessing import MinMaxScaler
import time


class StockDataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def get_stock_data(self, symbol, period="1y", start=None, end=None):
        """Helper function to get stock data with rate limit handling"""
        # Add random delay between 1-3 seconds to avoid hitting rate limits
        time.sleep(np.random.uniform(1, 3))
        
        try:
            # Use either period or start/end dates
            if start and end:
                data = yf.download(symbol, start=start, end=end, progress=False)
            else:
                data = yf.download(symbol, period=period, progress=False)

            if data is None or data.empty:
                print(f"No data received for {symbol}")
                return None
                
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None
            
    def add_technical_indicators(self, data):
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
            
    def normalize_data(self, data):
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
    
    def prepare_data(self, symbol, period="1y", start=None, end=None):
        """Prepare stock data for prediction"""
        try:
            # Get data
            data = self.get_stock_data(symbol, period, start, end)
            
            if data is None or data.empty:
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
            print(f"Error preparing data for {symbol}: {str(e)}")
            return None, None


class DQNModel:
    def __init__(self, state_dim=13, action_dim=3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self._build_model()
        
    def _build_model(self):
        """Build a neural network model for deep Q-learning"""
        model = Sequential()
        model.add(Input(shape=(self.state_dim,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_dim, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model
    
    def load_weights(self, weights_path):
        """Load model weights from file"""
        try:
            self.model.load_weights(weights_path)
            print(f"Successfully loaded DQN weights from {weights_path}")
            return True
        except Exception as e:
            print(f"Error loading DQN weights: {e}")
            return False
    
    def predict(self, state):
        """Predict action Q-values for a state"""
        if not isinstance(state, np.ndarray):
            state = np.array(state)
            
        # Reshape state to match model input
        state = np.reshape(state, [1, self.state_dim])
        
        # Get Q-values for all actions
        q_values = self.model.predict(state, verbose=0)[0]
        
        # Get best action
        best_action = np.argmax(q_values)
        
        return best_action, q_values


class PPOModel:
    def __init__(self, state_dim=13, action_dim=3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
    def _build_actor(self):
        """Build the actor (policy) network"""
        inputs = Input(shape=(self.state_dim,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(self.action_dim, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.0001))
        return model
    
    def _build_critic(self):
        """Build the critic (value) network"""
        inputs = Input(shape=(self.state_dim,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(1, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model
    
    def load_weights(self, actor_path, critic_path):
        """Load actor and critic weights from files"""
        try:
            self.actor.load_weights(actor_path)
            print(f"Successfully loaded PPO actor weights from {actor_path}")
            self.critic.load_weights(critic_path)
            print(f"Successfully loaded PPO critic weights from {critic_path}")
            return True
        except Exception as e:
            print(f"Error loading PPO weights: {e}")
            return False
    
    def predict(self, state):
        """Predict action probabilities for a state"""
        if not isinstance(state, np.ndarray):
            state = np.array(state)
            
        # Reshape state to match model input
        state = np.reshape(state, [1, self.state_dim])
        
        # Get action probabilities from the actor
        action_probs = self.actor.predict(state, verbose=0)[0]
        
        # Get value estimate from the critic
        value = self.critic.predict(state, verbose=0)[0, 0]
        
        # Get best action (highest probability)
        best_action = np.argmax(action_probs)
        
        return best_action, action_probs, value


class TradingSimulator:
    def __init__(self, initial_balance=10000, transaction_fee_percent=0.001):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.portfolio = {}  # {symbol: {'shares': 0, 'avg_buy_price': 0}}
        self.transaction_history = []
        
    def reset(self):
        """Reset the simulator to its initial state"""
        self.balance = self.initial_balance
        self.portfolio = {}
        self.transaction_history = []
        
    def get_portfolio_value(self, current_prices):
        """Calculate total portfolio value"""
        portfolio_value = self.balance
        
        for symbol, position in self.portfolio.items():
            if symbol in current_prices:
                portfolio_value += position['shares'] * current_prices[symbol]
                
        return portfolio_value
    
    def get_position_info(self, symbol, current_price):
        """Get information about a position"""
        if symbol not in self.portfolio:
            return {'shares': 0, 'position_value': 0, 'cost_basis': 0, 'unrealized_pnl': 0, 'unrealized_pnl_percent': 0}
            
        position = self.portfolio[symbol]
        position_value = position['shares'] * current_price
        cost_basis = position['shares'] * position['avg_buy_price']
        unrealized_pnl = position_value - cost_basis
        unrealized_pnl_percent = (unrealized_pnl / cost_basis) * 100 if cost_basis > 0 else 0
        
        return {
            'shares': position['shares'],
            'position_value': position_value,
            'cost_basis': cost_basis,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_percent': unrealized_pnl_percent
        }
    
    def execute_trade(self, symbol, action, current_price, timestamp, investment_amount=None):
        """
        Execute a trade based on the predicted action
        
        Args:
            symbol: Stock symbol
            action: 0 (sell), 1 (hold), 2 (buy)
            current_price: Current stock price
            timestamp: Timestamp for the trade
            investment_amount: Optional amount to invest (for buy orders only)
            
        Returns:
            Dictionary with trade details
        """
        result = {
            'symbol': symbol,
            'action': action,
            'action_name': ['Sell', 'Hold', 'Buy'][action],
            'price': current_price,
            'timestamp': timestamp,
            'success': False,
            'shares_traded': 0,
            'trade_value': 0,
            'transaction_fee': 0,
            'balance_before': self.balance,
            'balance_after': self.balance
        }
        
        # Initialize portfolio entry if it doesn't exist
        if symbol not in self.portfolio:
            self.portfolio[symbol] = {'shares': 0, 'avg_buy_price': 0}
            
        # SELL
        if action == 0 and self.portfolio[symbol]['shares'] > 0:
            shares_to_sell = self.portfolio[symbol]['shares']
            sale_value = shares_to_sell * current_price
            transaction_fee = sale_value * self.transaction_fee_percent
            
            self.balance += (sale_value - transaction_fee)
            
            result['success'] = True
            result['shares_traded'] = shares_to_sell
            result['trade_value'] = sale_value
            result['transaction_fee'] = transaction_fee
            result['balance_after'] = self.balance
            
            # Record transaction
            self.transaction_history.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'action': 'SELL',
                'price': current_price,
                'shares': shares_to_sell,
                'value': sale_value,
                'fee': transaction_fee,
                'balance': self.balance
            })
            
            # Update portfolio
            self.portfolio[symbol]['shares'] = 0
            self.portfolio[symbol]['avg_buy_price'] = 0
            
        # BUY
        elif action == 2 and self.balance > 0:
            if investment_amount is not None and investment_amount > 0 and investment_amount <= self.balance:
                # Use specified investment amount
                amount_to_invest = investment_amount
            else:
                # Use 95% of available balance by default
                amount_to_invest = self.balance * 0.95
                
            # Calculate shares to buy after considering transaction fee
            amount_after_fee = amount_to_invest / (1 + self.transaction_fee_percent)
            shares_to_buy = int(amount_after_fee / current_price)
            
            if shares_to_buy > 0:
                purchase_value = shares_to_buy * current_price
                transaction_fee = purchase_value * self.transaction_fee_percent
                total_cost = purchase_value + transaction_fee
                
                self.balance -= total_cost
                
                # Update average purchase price
                total_shares = self.portfolio[symbol]['shares'] + shares_to_buy
                total_cost_basis = (self.portfolio[symbol]['shares'] * self.portfolio[symbol]['avg_buy_price']) + purchase_value
                new_avg_price = total_cost_basis / total_shares if total_shares > 0 else 0
                
                self.portfolio[symbol]['shares'] = total_shares
                self.portfolio[symbol]['avg_buy_price'] = new_avg_price
                
                result['success'] = True
                result['shares_traded'] = shares_to_buy
                result['trade_value'] = purchase_value
                result['transaction_fee'] = transaction_fee
                result['balance_after'] = self.balance
                
                # Record transaction
                self.transaction_history.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': 'BUY',
                    'price': current_price,
                    'shares': shares_to_buy,
                    'value': purchase_value,
                    'fee': transaction_fee,
                    'balance': self.balance
                })
        
        return result


class MultiModelTradingStrategy:
    def __init__(self, models_dir, model_type="dqn", initial_balance=10000):
        self.models_dir = models_dir
        self.model_type = model_type.lower()  # "dqn" or "ppo"
        self.initial_balance = initial_balance
        self.models = {}
        self.preprocessors = {}
        self.simulator = TradingSimulator(initial_balance=initial_balance)
        
    def load_models(self):
        """Load all model weights from the models directory"""
        if self.model_type == "dqn":
            # Find all DQN model weights
            weight_files = glob.glob(os.path.join(self.models_dir, "*_DQN_*.weights.h5"))
            
            for weight_file in weight_files:
                try:
                    # Extract symbol from filename (assuming format: SYMBOL_DQN_*)
                    filename = os.path.basename(weight_file)
                    symbol = filename.split('_')[0]
                    
                    if symbol not in self.models:
                        # Create model
                        self.models[symbol] = DQNModel()
                        self.preprocessors[symbol] = StockDataPreprocessor()
                    
                    # Load weights
                    self.models[symbol].load_weights(weight_file)
                    print(f"Loaded DQN model for {symbol} from {weight_file}")
                except Exception as e:
                    print(f"Error loading model from {weight_file}: {str(e)}")
        
        elif self.model_type == "ppo":
            # Find all PPO actor model weights
            actor_files = glob.glob(os.path.join(self.models_dir, "*_PPO_actor_*.weights.h5"))
            
            for actor_file in actor_files:
                try:
                    # Extract symbol from filename (assuming format: SYMBOL_PPO_actor_*)
                    filename = os.path.basename(actor_file)
                    symbol = filename.split('_')[0]
                    
                    # Determine corresponding critic file
                    if "best" in actor_file:
                        critic_file = actor_file.replace("actor_best", "critic_best")
                    else:
                        critic_file = actor_file.replace("actor_final", "critic_final")
                    
                    if not os.path.exists(critic_file):
                        print(f"Warning: Critic file {critic_file} not found for {symbol}")
                        continue
                    
                    if symbol not in self.models:
                        # Create model
                        self.models[symbol] = PPOModel()
                        self.preprocessors[symbol] = StockDataPreprocessor()
                    
                    # Load weights
                    self.models[symbol].load_weights(actor_file, critic_file)
                    print(f"Loaded PPO model for {symbol} from {actor_file} and {critic_file}")
                except Exception as e:
                    print(f"Error loading model from {actor_file}: {str(e)}")
        
        return len(self.models) > 0
    
    def get_available_symbols(self):
        """Get list of symbols with loaded models"""
        return list(self.models.keys())
    
    def create_state(self, data, idx, balance, shares, portfolio_value):
        """Create a state representation for the model input"""
        # Get the current market data
        market_data = data.iloc[idx].values
        
        # Portfolio information (normalized)
        portfolio_info = np.array([
            balance / self.initial_balance,  # Normalized balance
            shares * data['Close'].iloc[idx] / self.initial_balance,  # Normalized position value
            portfolio_value / self.initial_balance  # Normalized portfolio value
        ])
        
        # Combine market data and portfolio info
        state = np.concatenate([market_data, portfolio_info])
        
        return state
    
    def get_prediction(self, symbol, state):
        """Get a model prediction for a given state"""
        if symbol not in self.models:
            print(f"No model found for {symbol}")
            return 1, None  # Default to HOLD
            
        model = self.models[symbol]
        
        if self.model_type == "dqn":
            action, q_values = model.predict(state)
            prediction_details = {
                'q_values': q_values,
                'q_sell': q_values[0],
                'q_hold': q_values[1],
                'q_buy': q_values[2]
            }
        else:  # PPO
            action, action_probs, value = model.predict(state)
            prediction_details = {
                'action_probs': action_probs,
                'prob_sell': action_probs[0],
                'prob_hold': action_probs[1],
                'prob_buy': action_probs[2],
                'value': value
            }
            
        return action, prediction_details
    
    def backtest(self, symbols=None, period="3mo", start=None, end=None):
        """Backtest the trading strategy on historical data"""
        if not self.models:
            print("No models loaded. Please load models first using load_models()")
            return None
            
        if symbols is None:
            symbols = self.get_available_symbols()
        elif isinstance(symbols, str):
            symbols = [symbols]
            
        # Reset the simulator
        self.simulator.reset()
        
        # Store results for each symbol
        results = {}
        
        # Track overall portfolio performance
        portfolio_values = []
        dates = []
        
        # Process each symbol
        for symbol in symbols:
            if symbol not in self.models:
                print(f"No model found for {symbol}. Skipping.")
                continue
                
            # Prepare data
            preprocessor = self.preprocessors[symbol]
            normalized_data, original_data = preprocessor.prepare_data(symbol, period, start, end)
            
            if normalized_data is None or original_data is None:
                print(f"Could not prepare data for {symbol}. Skipping.")
                continue
                
            # Track predictions and trades for this symbol
            symbol_results = {
                'dates': original_data.index,
                'prices': original_data['Close'].values,
                'predictions': [],
                'actions': [],
                'trades': [],
                'portfolio_values': []
            }
            
            # Initialize position info
            shares_held = 0
            
            # For each day in the data
            for i in range(len(normalized_data)):
                # Get current price and date
                current_price = original_data['Close'].iloc[i]
                current_date = original_data.index[i]
                
                # Get total portfolio value including cash and all positions
                current_prices = {s: original_data['Close'].iloc[i] for s in symbols if s == symbol}
                portfolio_value = self.simulator.get_portfolio_value(current_prices)
                
                # Create state representation
                state = self.create_state(normalized_data, i, 
                                         self.simulator.balance, 
                                         shares_held, 
                                         portfolio_value)
                
                # Get model prediction
                action, prediction_details = self.get_prediction(symbol, state)
                
                # Execute trade based on prediction
                trade_result = self.simulator.execute_trade(
                    symbol, action, current_price, current_date
                )
                
                # Update shares held after trade
                shares_held = self.simulator.portfolio.get(symbol, {'shares': 0})['shares']
                
                # Record results
                symbol_results['predictions'].append(prediction_details)
                symbol_results['actions'].append(action)
                
                if trade_result['success']:
                    symbol_results['trades'].append(trade_result)
                
                # Update portfolio values
                current_prices = {s: original_data['Close'].iloc[i] for s in symbols if s == symbol}
                updated_portfolio_value = self.simulator.get_portfolio_value(current_prices)
                symbol_results['portfolio_values'].append(updated_portfolio_value)
                
                # Track overall portfolio values and dates (only from the first symbol to avoid duplicates)
                if symbol == symbols[0]:
                    portfolio_values.append(updated_portfolio_value)
                    dates.append(current_date)
            
            # Calculate performance metrics for this symbol
            initial_price = symbol_results['prices'][0]
            final_price = symbol_results['prices'][-1]
            buy_hold_return = (final_price / initial_price - 1) * 100
            
            initial_value = self.initial_balance
            final_value = symbol_results['portfolio_values'][-1] if symbol_results['portfolio_values'] else self.initial_balance
            strategy_return = (final_value / initial_value - 1) * 100
            
            symbol_results['metrics'] = {
                'initial_price': initial_price,
                'final_price': final_price,
                'buy_hold_return': buy_hold_return,
                'initial_portfolio_value': initial_value,
                'final_portfolio_value': final_value,
                'strategy_return': strategy_return,
                'outperformance': strategy_return - buy_hold_return
            }
            
            results[symbol] = symbol_results
        
        # Calculate overall portfolio metrics
        overall_metrics = {
            'initial_value': self.initial_balance,
            'final_value': portfolio_values[-1] if portfolio_values else self.initial_balance,
            'return': ((portfolio_values[-1] / self.initial_balance - 1) * 100) if portfolio_values else 0,
            'transaction_count': len(self.simulator.transaction_history)
        }
        
        # Add overall portfolio data
        results['portfolio'] = {
            'dates': dates,
            'values': portfolio_values,
            'transactions': self.simulator.transaction_history,
            'metrics': overall_metrics
        }
        
        return results
    
    def predict_current_positions(self, symbols=None, period="1mo"):
        """Get predictions for current positions based on recent data"""
        if not self.models:
            print("No models loaded. Please load models first using load_models()")
            return None
            
        if symbols is None:
            symbols = self.get_available_symbols()
        elif isinstance(symbols, str):
            symbols = [symbols]
            
        # Get current predictions
        predictions = {}
        
        for symbol in symbols:
            if symbol not in self.models:
                print(f"No model found for {symbol}. Skipping.")
                continue
                
            # Prepare data
            preprocessor = self.preprocessors[symbol]
            normalized_data, original_data = preprocessor.prepare_data(symbol, period)
            
            if normalized_data is None or original_data is None:
                print(f"Could not prepare data for {symbol}. Skipping.")
                continue
                
            # Get the most recent state
            latest_idx = len(normalized_data) - 1
            latest_price = original_data['Close'].iloc[latest_idx]
            latest_date = original_data.index[latest_idx]
            
            # Get position info for this symbol
            position_info = self.simulator.get_position_info(symbol, latest_price)
            shares_held = position_info['shares']
            
            # Get total portfolio value
            current_prices = {s: original_data['Close'].iloc[latest_idx] for s in symbols if s == symbol}
            portfolio_value = self.simulator.get_portfolio_value(current_prices)
            
            # Create state representation
            state = self.create_state(normalized_data, latest_idx, 
                                     self.simulator.balance, 
                                     shares_held, 
                                     portfolio_value)
            
            # Get prediction
            action, prediction_details = self.get_prediction(symbol, state)
            
            # Record prediction
            predictions[symbol] = {
                'symbol': symbol,
                'date': latest_date,
                'price': latest_price,
                'action': action,
                'action_name': ['Sell', 'Hold', 'Buy'][action],
                'prediction_details': prediction_details,
                'position': position_info
            }
        
        return predictions
    
    def visualize_backtest_results(self, results, save_path=None):
        """Visualize backtest results"""
        if not results:
            print("No results to visualize")
            return
            
        # Extract portfolio data
        portfolio_data = results.get('portfolio', {})
        dates = portfolio_data.get('dates', [])
        portfolio_values = portfolio_data.get('values', [])
        
        if not dates or not portfolio_values:
            print("No portfolio data to visualize")
            return
            
        # Create figure with three subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [2, 1, 2]})
        
        # Plot 1: Portfolio Value
        axs[0].plot(dates, portfolio_values, label='Portfolio Value', color='blue')
        axs[0].set_title('Portfolio Value Over Time')
        axs[0].set_xlabel('Date')
        axs[0].set_ylabel('Value ($)')
        axs[0].grid(True)
        axs[0].legend()
        
        # Plot 2: Transactions
        transactions = portfolio_data.get('transactions', [])
        
        if transactions:
            buy_dates = [t['timestamp'] for t in transactions if t['action'] == 'BUY']
            buy_values = [t['value'] for t in transactions if t['action'] == 'BUY']
            
            sell_dates = [t['timestamp'] for t in transactions if t['action'] == 'SELL']
            sell_values = [t['value'] for t in transactions if t['action'] == 'SELL']
            
            axs[1].scatter(buy_dates, buy_values, color='green', label='Buy', marker='^', s=100)
            axs[1].scatter(sell_dates, sell_values, color='red', label='Sell', marker='v', s=100)
            axs[1].set_title('Transactions')
            axs[1].set_xlabel('Date')
            axs[1].set_ylabel('Transaction Value ($)')
            axs[1].grid(True)
            axs[1].legend()
        else:
            axs[1].text(0.5, 0.5, 'No transactions', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)
        
        # Plot 3: Individual Stock Performance
        symbols = [s for s in results.keys() if s != 'portfolio']
        
        for symbol in symbols:
            symbol_data = results[symbol]
            symbol_dates = symbol_data['dates']
            symbol_prices = symbol_data['prices']
            
            # Normalize prices for comparison
            normalized_prices = [p / symbol_prices[0] for p in symbol_prices]
            
            axs[2].plot(symbol_dates, normalized_prices, label=f'{symbol} (Price)')
        
        # Add normalized portfolio value for comparison
        if portfolio_values:
            normalized_portfolio = [v / portfolio_values[0] for v in portfolio_values]
            axs[2].plot(dates, normalized_portfolio, label='Portfolio (Value)', linewidth=2, color='black')
        
        axs[2].set_title('Normalized Performance Comparison')
        axs[2].set_xlabel('Date')
        axs[2].set_ylabel('Normalized Value')
        axs[2].grid(True)
        axs[2].legend()
        
        # Add metrics as text
        overall_metrics = portfolio_data.get('metrics', {})
        metrics_text = (
            f"Initial Value: ${overall_metrics.get('initial_value', 0):.2f}\n"
            f"Final Value: ${overall_metrics.get('final_value', 0):.2f}\n"
            f"Return: {overall_metrics.get('return', 0):.2f}%\n"
            f"Transaction Count: {overall_metrics.get('transaction_count', 0)}"
        )
        plt.figtext(0.5, 0.01, metrics_text, ha='center', bbox={'facecolor': 'lightblue', 'alpha': 0.5, 'pad': 5})
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Results visualization saved to {save_path}")
        
        plt.show()
        
    def visualize_symbol_predictions(self, symbol, results, save_path=None):
        """Visualize prediction results for a specific symbol"""
        if symbol not in results:
            print(f"No results found for {symbol}")
            return
            
        symbol_data = results[symbol]
        
        # Create figure with three subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot 1: Stock Price with Buy/Sell Signals
        axs[0].plot(symbol_data['dates'], symbol_data['prices'], label=f'{symbol} Price', color='blue')
        
        # Add buy/sell markers
        for trade in symbol_data.get('trades', []):
            if trade['action'] == 0:  # Sell
                axs[0].scatter(trade['timestamp'], trade['price'], color='red', marker='v', s=100, label='_nolegend_')
            elif trade['action'] == 2:  # Buy
                axs[0].scatter(trade['timestamp'], trade['price'], color='green', marker='^', s=100, label='_nolegend_')
        
        axs[0].set_title(f'{symbol} Price with Trading Signals')
        axs[0].set_xlabel('Date')
        axs[0].set_ylabel('Price ($)')
        axs[0].grid(True)
        
        # Add custom legend
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color='blue', lw=2),
            Line2D([0], [0], marker='^', color='green', markersize=10, linestyle='None'),
            Line2D([0], [0], marker='v', color='red', markersize=10, linestyle='None')
        ]
        axs[0].legend(custom_lines, [f'{symbol} Price', 'Buy Signal', 'Sell Signal'])
        
        # Plot 2: Model Predictions
        if self.model_type == "dqn":
            # Extract Q-values
            q_sell = [pred.get('q_sell', 0) for pred in symbol_data.get('predictions', [])]
            q_hold = [pred.get('q_hold', 0) for pred in symbol_data.get('predictions', [])]
            q_buy = [pred.get('q_buy', 0) for pred in symbol_data.get('predictions', [])]
            
            axs[1].plot(symbol_data['dates'], q_sell, label='Q(Sell)', color='red')
            axs[1].plot(symbol_data['dates'], q_hold, label='Q(Hold)', color='gray')
            axs[1].plot(symbol_data['dates'], q_buy, label='Q(Buy)', color='green')
        else:  # PPO
            # Extract action probabilities
            prob_sell = [pred.get('prob_sell', 0) for pred in symbol_data.get('predictions', [])]
            prob_hold = [pred.get('prob_hold', 0) for pred in symbol_data.get('predictions', [])]
            prob_buy = [pred.get('prob_buy', 0) for pred in symbol_data.get('predictions', [])]
            
            axs[1].plot(symbol_data['dates'], prob_sell, label='P(Sell)', color='red')
            axs[1].plot(symbol_data['dates'], prob_hold, label='P(Hold)', color='gray')
            axs[1].plot(symbol_data['dates'], prob_buy, label='P(Buy)', color='green')
        
        axs[1].set_title(f'Model Predictions for {symbol}')
        axs[1].set_xlabel('Date')
        axs[1].set_ylabel('Prediction Values')
        axs[1].grid(True)
        axs[1].legend()
        
        # Plot 3: Actions and Portfolio Value
        actions = symbol_data.get('actions', [])
        dates = symbol_data['dates']
        
        colors = ['red' if a == 0 else 'gray' if a == 1 else 'green' for a in actions]
        
        axs[2].scatter(dates, [0] * len(dates), c=colors, marker='o', s=50)
        axs[2].set_yticks([-1, 0, 1])
        axs[2].set_yticklabels(['', 'Actions', ''])
        
        # Add portfolio value on second y-axis
        ax_twin = axs[2].twinx()
        ax_twin.plot(dates, symbol_data.get('portfolio_values', []), color='blue', label='Portfolio Value')
        ax_twin.set_ylabel('Portfolio Value ($)')
        
        axs[2].set_title(f'Actions and Portfolio Value for {symbol}')
        axs[2].set_xlabel('Date')
        
        # Add custom legend for actions
        from matplotlib.patches import Patch
        custom_patches = [
            Patch(color='red', label='Sell'),
            Patch(color='gray', label='Hold'),
            Patch(color='green', label='Buy'),
            Line2D([0], [0], color='blue', lw=2, label='Portfolio Value')
        ]
        axs[2].legend(handles=custom_patches, loc='upper left')
        
        # Add metrics as text
        metrics = symbol_data.get('metrics', {})
        metrics_text = (
            f"Initial Price: ${metrics.get('initial_price', 0):.2f}\n"
            f"Final Price: ${metrics.get('final_price', 0):.2f}\n"
            f"Buy & Hold Return: {metrics.get('buy_hold_return', 0):.2f}%\n"
            f"Strategy Return: {metrics.get('strategy_return', 0):.2f}%\n"
            f"Outperformance: {metrics.get('outperformance', 0):.2f}%"
        )
        plt.figtext(0.5, 0.01, metrics_text, ha='center', bbox={'facecolor': 'lightblue', 'alpha': 0.5, 'pad': 5})
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Symbol predictions visualization saved to {save_path}")
        
        plt.show()


def create_state_from_current_data(symbol, model_type):
    """
    Helper function to create a state from the most recent data for a single prediction
    
    Returns:
        tuple: (state, price, date)
    """
    preprocessor = StockDataPreprocessor()
    normalized_data, original_data = preprocessor.prepare_data(symbol, period="1mo")
    
    if normalized_data is None or original_data is None:
        print(f"Could not prepare data for {symbol}")
        return None, None, None
    
    # Get the most recent data point
    latest_idx = len(normalized_data) - 1
    latest_price = original_data['Close'].iloc[latest_idx]
    latest_date = original_data.index[latest_idx]
    
    # Create a state
    # Market data
    market_data = normalized_data.iloc[latest_idx].values
    
    # Portfolio info (assuming no existing position)
    portfolio_info = np.array([1.0, 0.0, 1.0])  # normalized balance=100%, shares=0%, portfolio=100%
    
    # Combine market data and portfolio info
    state = np.concatenate([market_data, portfolio_info])
    
    return state, latest_price, latest_date


def predict_single_symbol(symbol, model_path, model_type="dqn"):
    """Make a prediction for a single symbol using a specific model file"""
    # Prepare the state
    state, price, date = create_state_from_current_data(symbol, model_type)
    
    if state is None:
        return None
    
    # Load model and make prediction
    if model_type.lower() == "dqn":
        model = DQNModel()
        success = model.load_weights(model_path)
        
        if not success:
            return None
            
        action, q_values = model.predict(state)
        
        return {
            'symbol': symbol,
            'date': date,
            'price': price,
            'action': action,
            'action_name': ['Sell', 'Hold', 'Buy'][action],
            'q_values': q_values,
            'model_type': 'DQN',
            'model_path': model_path
        }
        
    elif model_type.lower() == "ppo":
        # For PPO, we need actor and critic paths
        if "actor" not in model_path:
            print("Error: For PPO models, specify the actor weights path")
            return None
            
        # Try to find the corresponding critic path
        critic_path = None
        if "actor_best" in model_path:
            critic_path = model_path.replace("actor_best", "critic_best")
        elif "actor_final" in model_path:
            critic_path = model_path.replace("actor_final", "critic_final")
            
        if critic_path is None or not os.path.exists(critic_path):
            print(f"Error: Could not find critic weights at {critic_path}")
            return None
            
        model = PPOModel()
        success = model.load_weights(model_path, critic_path)
        
        if not success:
            return None
            
        action, action_probs, value = model.predict(state)
        
        return {
            'symbol': symbol,
            'date': date,
            'price': price,
            'action': action,
            'action_name': ['Sell', 'Hold', 'Buy'][action],
            'action_probs': action_probs,
            'value': value,
            'model_type': 'PPO',
            'model_path': model_path
        }
    
    else:
        print(f"Unsupported model type: {model_type}")
        return None


def run_strategy_demo():
    """Run a demo of the multi-model trading strategy"""
    # Configuration
    models_dir = "models"  # Change to your models directory
    model_type = "dqn"  # or "ppo"
    initial_balance = 10000
    backtest_period = "6mo"
    
    # Create and load the strategy
    strategy = MultiModelTradingStrategy(models_dir, model_type, initial_balance)
    loaded = strategy.load_models()
    
    if not loaded:
        print("No models were loaded. Please check the models directory.")
        return
    
    # Print available symbols
    symbols = strategy.get_available_symbols()
    print(f"Loaded models for {len(symbols)} symbols: {', '.join(symbols)}")
    
    # Run backtest
    print(f"\nRunning backtest with {model_type.upper()} models for the past {backtest_period}...")
    results = strategy.backtest(symbols, period=backtest_period)
    
    if results:
        # Display overall performance
        portfolio_metrics = results['portfolio']['metrics']
        print("\nBacktest Results:")
        print(f"Initial Portfolio Value: ${portfolio_metrics['initial_value']:.2f}")
        print(f"Final Portfolio Value: ${portfolio_metrics['final_value']:.2f}")
        print(f"Return: {portfolio_metrics['return']:.2f}%")
        print(f"Number of Transactions: {portfolio_metrics['transaction_count']}")
        
        # Display individual symbol metrics
        print("\nPerformance by Symbol:")
        for symbol in symbols:
            if symbol in results:
                metrics = results[symbol]['metrics']
                print(f"{symbol}: Strategy Return = {metrics['strategy_return']:.2f}%, "
                      f"Buy & Hold Return = {metrics['buy_hold_return']:.2f}%, "
                      f"Outperformance = {metrics['outperformance']:.2f}%")
        
        # Visualize results
        strategy.visualize_backtest_results(results, save_path=f"{model_type}_backtest_results.png")
        
        # Visualize individual symbol predictions
        for symbol in symbols[:1]:  # Just visualize the first symbol to avoid too many plots
            strategy.visualize_symbol_predictions(symbol, results, save_path=f"{symbol}_{model_type}_predictions.png")
    
    # Get current predictions
    print("\nCurrent Trading Signals:")
    predictions = strategy.predict_current_positions(symbols)
    
    if predictions:
        for symbol, prediction in predictions.items():
            print(f"{symbol}: Current Price = ${prediction['price']:.2f}, "
                  f"Action = {prediction['action_name']}")
    else:
        print("No predictions available.")


if __name__ == "__main__":
    run_strategy_demo()