import tensorflow as tf
import numpy as np
import pandas as pd
import yfinance as yf
import os
import glob
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import argparse
from sklearn.preprocessing import MinMaxScaler
import time


class StockDataProcessor:
    """Process stock data and create inputs for model prediction"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def get_stock_data(self, symbol, period="1mo"):
        """Download stock data from Yahoo Finance"""
        try:
            data = yf.download(symbol, period=period, progress=False)
            if data.empty:
                print(f"No data available for {symbol}")
                return None
            return data
        except Exception as e:
            print(f"Error downloading data for {symbol}: {e}")
            return None
    
    def add_indicators(self, data):
        """Add technical indicators to the data"""
        df = data.copy()
        
        # Add moving averages
        df['MA5'] = df['Close'].rolling(window=5, min_periods=1).mean()
        df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        
        # Add RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        
        # Avoid division by zero
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs.fillna(0)))
        
        # Fill any NaN values that might have been created
        df = df.ffill().bfill()  # Using ffill and bfill instead of fillna(method='ffill')
        
        return df
    
    def normalize_data(self, data):
        """Normalize the data for model input"""
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'RSI']
        
        normalized = data[features].copy()
        
        # Log transform volume
        normalized['Volume'] = np.log1p(normalized['Volume'])
        
        # Scale features
        for feature in features:
            if feature != 'Volume':  # Volume already transformed
                feature_data = normalized[feature].values.reshape(-1, 1)
                normalized[feature] = self.scaler.fit_transform(feature_data)
        
        return normalized
    
    def prepare_data(self, symbol, period="1mo"):
        """Prepare data for model prediction"""
        # Get stock data
        data = self.get_stock_data(symbol, period)
        if data is None:
            return None, None
        
        # Add technical indicators
        data_with_indicators = self.add_indicators(data)
        
        # Normalize the data
        normalized_data = self.normalize_data(data_with_indicators)
        
        return normalized_data, data_with_indicators
    
    def create_state(self, normalized_data, index, portfolio_info=None):
        """Create state representation for model input"""
        # Get market data
        market_data = normalized_data.iloc[index].values
        
        # If portfolio info not provided, use default values
        if portfolio_info is None:
            # Default values: normalized balance=1.0, shares_value=0.0, portfolio_value=1.0
            portfolio_info = np.array([1.0, 0.0, 1.0])
        
        # Combine market data and portfolio info
        state = np.concatenate([market_data, portfolio_info])
        
        # Pad the state to have 27 dimensions as required by the saved model
        if len(state) < 27:
            padding = np.zeros(27 - len(state))
            state = np.concatenate([state, padding])
        
        return state


class DQNModel:
    """DQN model for predicting trading actions"""
    
    def __init__(self, state_dim=27, action_dim=3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self._build_model()
    
    def _build_model(self):
        """Build a neural network model for DQN"""
        model = Sequential()
        model.add(Input(shape=(self.state_dim,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_dim, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model
    
    def load_weights(self, weights_file):
        """Load model weights from file"""
        try:
            self.model.load_weights(weights_file)
            print(f"Loaded DQN weights from {weights_file}")
            return True
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False
    
    def predict(self, state):
        """Predict action for a given state"""
        # Ensure state is a numpy array
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        
        # Reshape for model input
        state = np.reshape(state, [1, self.state_dim])
        
        # Get Q-values
        q_values = self.model.predict(state, verbose=0)[0]
        
        # Get best action
        action = np.argmax(q_values)
        
        return action, q_values


class PPOModel:
    """PPO model for predicting trading actions"""
    
    def __init__(self, state_dim=27, action_dim=3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = self._build_actor()
        self.critic = self._build_critic()
    
    def _build_actor(self):
        """Build actor network"""
        inputs = Input(shape=(self.state_dim,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(self.action_dim, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.0001))
        return model
    
    def _build_critic(self):
        """Build critic network"""
        inputs = Input(shape=(self.state_dim,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(1, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model
    
    def load_weights(self, actor_weights, critic_weights):
        """Load actor and critic weights"""
        try:
            self.actor.load_weights(actor_weights)
            print(f"Loaded PPO actor weights from {actor_weights}")
            
            self.critic.load_weights(critic_weights)
            print(f"Loaded PPO critic weights from {critic_weights}")
            
            return True
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False
    
    def predict(self, state):
        """Predict action for a given state"""
        # Ensure state is a numpy array
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        
        # Reshape for model input
        state = np.reshape(state, [1, self.state_dim])
        
        # Get action probabilities from actor
        action_probs = self.actor.predict(state, verbose=0)[0]
        
        # Get value from critic
        value = self.critic.predict(state, verbose=0)[0, 0]
        
        # Get best action
        action = np.argmax(action_probs)
        
        return action, action_probs, value


def find_model_files(models_dir):
    """Find and organize all model files in the directory"""
    model_files = {}
    
    # Find DQN model files
    dqn_files = glob.glob(os.path.join(models_dir, "*_DQN_*.weights.h5"))
    for file in dqn_files:
        filename = os.path.basename(file)
        parts = filename.split('_')
        if len(parts) >= 3:
            symbol = parts[0]
            if symbol not in model_files:
                model_files[symbol] = {'dqn': []}
            model_files[symbol]['dqn'].append(file)
    
    # Find PPO model files
    ppo_actor_files = glob.glob(os.path.join(models_dir, "*_PPO_actor_*.weights.h5"))
    for actor_file in ppo_actor_files:
        filename = os.path.basename(actor_file)
        parts = filename.split('_')
        if len(parts) >= 4:
            symbol = parts[0]
            
            # Find matching critic file
            if "best" in actor_file:
                critic_file = actor_file.replace("actor_best", "critic_best")
            else:
                critic_file = actor_file.replace("actor_final", "critic_final")
            
            if os.path.exists(critic_file):
                if symbol not in model_files:
                    model_files[symbol] = {'ppo': []}
                elif 'ppo' not in model_files[symbol]:
                    model_files[symbol]['ppo'] = []
                
                model_files[symbol]['ppo'].append((actor_file, critic_file))
    
    return model_files


def make_prediction(symbol, model_type, weight_files, data_processor=None):
    """Make a prediction for a symbol using the specified model"""
    # Create data processor if not provided
    if data_processor is None:
        data_processor = StockDataProcessor()
    
    # Prepare data
    normalized_data, original_data = data_processor.prepare_data(symbol)
    if normalized_data is None:
        print(f"Could not prepare data for {symbol}")
        return None
    
    # Get latest price and date
    latest_idx = len(normalized_data) - 1
    latest_price = original_data['Close'].iloc[latest_idx]
    latest_date = original_data.index[latest_idx]
    
    # Create state
    state = data_processor.create_state(normalized_data, latest_idx)
    
    # Initialize result dictionary
    result = {
        'symbol': symbol,
        'date': latest_date,
        'price': latest_price,
        'model_type': model_type
    }
    
    # Make prediction based on model type
    if model_type.lower() == 'dqn':
        # For DQN, use the last weight file (assuming it's the most recent/best)
        weight_file = weight_files[-1]
        model = DQNModel(state_dim=len(state))
        if model.load_weights(weight_file):
            action, q_values = model.predict(state)
            
            # Add prediction details to result
            result.update({
                'action': action,
                'action_name': ['Sell', 'Hold', 'Buy'][action],
                'q_values': q_values.tolist(),
                'weight_file': weight_file
            })
            
            return result
    
    elif model_type.lower() == 'ppo':
        # For PPO, use the last actor-critic pair
        actor_file, critic_file = weight_files[-1]
        model = PPOModel(state_dim=len(state))
        if model.load_weights(actor_file, critic_file):
            action, action_probs, value = model.predict(state)
            
            # Add prediction details to result
            result.update({
                'action': action,
                'action_name': ['Sell', 'Hold', 'Buy'][action],
                'action_probs': action_probs.tolist(),
                'value': float(value),
                'actor_file': actor_file,
                'critic_file': critic_file
            })
            
            return result
    
    return None


def predict_all_models(models_dir):
    """Make predictions using all available models"""
    # Find model files
    model_files = find_model_files(models_dir)
    
    if not model_files:
        print(f"No model files found in {models_dir}")
        return []
    
    # Create data processor
    data_processor = StockDataProcessor()
    
    # Make predictions for each symbol and model type
    predictions = []
    
    for symbol, models in model_files.items():
        print(f"\nProcessing {symbol}...")
        
        for model_type, weight_files in models.items():
            if weight_files:
                prediction = make_prediction(symbol, model_type, weight_files, data_processor)
                if prediction:
                    predictions.append(prediction)
                    
                    # Print prediction
                    if model_type.lower() == 'dqn':
                        print(f"  DQN predicts: {prediction['action_name']} "
                              f"(Q-values: Sell={prediction['q_values'][0]:.4f}, "
                              f"Hold={prediction['q_values'][1]:.4f}, "
                              f"Buy={prediction['q_values'][2]:.4f})")
                    else:  # PPO
                        print(f"  PPO predicts: {prediction['action_name']} "
                              f"(Probabilities: Sell={prediction['action_probs'][0]:.4f}, "
                              f"Hold={prediction['action_probs'][1]:.4f}, "
                              f"Buy={prediction['action_probs'][2]:.4f})")
    
    return predictions

def visualize_predictions(predictions):
    """Visualize the predictions"""
    if not predictions:
        print("No predictions to visualize")
        return
    
    # Group predictions by symbol
    symbol_predictions = {}
    for p in predictions:
        symbol = p['symbol']
        if symbol not in symbol_predictions:
            symbol_predictions[symbol] = []
        symbol_predictions[symbol].append(p)
    
    # Create figure with subplots
    num_symbols = len(symbol_predictions)
    fig, axs = plt.subplots(num_symbols, 1, figsize=(10, 5 * num_symbols))
    
    # Handle case with only one symbol
    if num_symbols == 1:
        axs = [axs]
    
    # Plot predictions for each symbol
    for i, (symbol, preds) in enumerate(symbol_predictions.items()):
        ax = axs[i]
        
        # Set up bar positions
        bar_width = 0.35
        positions = np.arange(len(preds))
        
        # Extract data
        model_types = [p['model_type'].upper() for p in preds]
        actions = [p['action_name'] for p in preds]
        
        # Get price (handle both float and pandas Series)
        price = preds[0]['price']
        if hasattr(price, 'iloc'):  # If it's a pandas Series
            price = price.iloc[0]
        
        # Create bars with colors based on action
        colors = ['red' if a == 'Sell' else 'gray' if a == 'Hold' else 'green' for a in actions]
        
        # Plot bars
        bars = ax.bar(positions, [1] * len(positions), bar_width, color=colors)
        
        # Add labels
        ax.set_title(f"{symbol} - Current Price: ${float(price):.2f}")
        ax.set_xticks(positions)
        ax.set_xticklabels(model_types)
        ax.set_yticks([])
        
        # Add action labels above bars
        for j, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width()/2, 0.5, actions[j],
                   ha='center', va='center', color='white', fontweight='bold')
        
        # Add detailed text below
        for j, p in enumerate(preds):
            if p['model_type'].lower() == 'dqn':
                detail_text = f"Q: {p['q_values'][0]:.2f}, {p['q_values'][1]:.2f}, {p['q_values'][2]:.2f}"
            else:  # PPO
                detail_text = f"P: {p['action_probs'][0]:.2f}, {p['action_probs'][1]:.2f}, {p['action_probs'][2]:.2f}"
            
            ax.text(j, -0.2, detail_text, ha='center', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("model_predictions.png")
    plt.show()

    
def main():
    parser = argparse.ArgumentParser(description='Make predictions using trained RL models')
    parser.add_argument('--models_dir', type=str, default='models', 
                        help='Directory containing model weight files')
    parser.add_argument('--symbol', type=str, 
                        help='Specific stock symbol to predict')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the predictions')
    
    args = parser.parse_args()
    
    # Find model files
    model_files = find_model_files(args.models_dir)
    
    if not model_files:
        print(f"No model files found in {args.models_dir}")
        return
    
    # Print available symbols and models
    print("Available models:")
    for symbol, models in model_files.items():
        model_types = list(models.keys())
        print(f"  {symbol}: {', '.join(model_types)}")
    
    # If symbol is specified, only predict for that symbol
    if args.symbol:
        if args.symbol in model_files:
            print(f"\nMaking predictions for {args.symbol}...")
            predictions = []
            for model_type, weight_files in model_files[args.symbol].items():
                if weight_files:
                    prediction = make_prediction(args.symbol, model_type, weight_files)
                    if prediction:
                        predictions.append(prediction)
        else:
            print(f"No models found for symbol {args.symbol}")
            return
    else:
        # Otherwise predict for all symbols
        print("\nMaking predictions for all symbols...")
        predictions = predict_all_models(args.models_dir)
    
    # Visualize predictions if requested
    if args.visualize and predictions:
        visualize_predictions(predictions)


def load_trained_model(model_path):
    """
    Load a trained model from the specified path.

    Args:
        model_path (str): Path to the trained model (.h5 file).

    Returns:
        model: Loaded Keras model.
    """
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model


if __name__ == "__main__":
    main()