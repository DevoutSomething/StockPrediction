"""
Configuration settings for the Stock Prediction API
"""
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent.parent / "logs" / "app.log", mode="a")
    ]
)
logger = logging.getLogger("stock-prediction")

# Environment configurations
class Config:
    """Configuration class for the application"""
    # Testing mode
    testing = os.environ.get("TESTING", "0") == "1"
    
    # API configuration
    api_config = {
        "host": os.environ.get("API_HOST", "0.0.0.0"),
        "port": int(os.environ.get("API_PORT", "8001")),
        "reload": os.environ.get("API_RELOAD", "0") == "1" or testing,
        "debug": os.environ.get("API_DEBUG", "0") == "1" or testing,
    }
    
    # ML configuration
    ml_config = {
        "default_algorithm": os.environ.get("ML_DEFAULT_ALGORITHM", "DQN"),
        "training_episodes": int(os.environ.get("ML_TRAINING_EPISODES", "20")),
        "use_mock_data": os.environ.get("USE_MOCK_DATA", "0") == "1" or testing,
    }
    
    # API keys
    api_keys = {
        "news_api": os.environ.get("NEWS_API_KEY", ""),
        "alpha_vantage": os.environ.get("ALPHA_VANTAGE_API_KEY", ""),
    }

# Create a global config instance
config = Config()

class config:
    """Application configuration manager"""
    
    def __init__(self, env_file: Optional[str] = None):
        # Base paths
        self.base_dir = Path(__file__).resolve().parent.parent
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.logs_dir = self.base_dir / "logs"
        
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Load environment variables from file if provided
        if env_file:
            self._load_env_file(env_file)
        
        # Database settings
        self.db_config = {
            "user": os.environ.get("DB_USER", "root"),
            "password": os.environ.get("DB_PASSWORD", "NewStrongPassword"),
            "host": os.environ.get("DB_HOST", "localhost"),
            "port": os.environ.get("DB_PORT", "3306"),
            "name": os.environ.get("DB_NAME", "stock_prediction"),
        }
        
        # API settings
        self.api_config = {
            "host": os.environ.get("API_HOST", "0.0.0.0"),
            "port": int(os.environ.get("API_PORT", "8000")),
            "debug": os.environ.get("API_DEBUG", "0") == "1",
            "reload": os.environ.get("API_RELOAD", "0") == "1",
        }
        
        # External API keys
        self.api_keys = {
            "news_api": os.environ.get("NEWS_API_KEY", ""),
            "alpha_vantage": os.environ.get("ALPHA_VANTAGE_API_KEY", ""),
        }
        
        # ML settings
        self.ml_config = {
            "default_algorithm": os.environ.get("ML_DEFAULT_ALGORITHM", "PPO"),
            "training_episodes": int(os.environ.get("ML_TRAINING_EPISODES", "50")),
            "use_mock_data": os.environ.get("USE_MOCK_DATA", "0") == "1",
            "model_cache_size": int(os.environ.get("MODEL_CACHE_SIZE", "10")),
        }
        
        # Testing mode
        self.testing = os.environ.get("TESTING", "0") == "1"
        if self.testing:
            self.db_config["name"] = "stock_prediction_test"
    
    def _load_env_file(self, env_file: str):
        """Load environment variables from a file"""
        try:
            env_path = Path(env_file)
            if not env_path.exists():
                print(f"Warning: Environment file {env_file} not found.")
                return
                
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"\'')
        except Exception as e:
            print(f"Error loading environment file: {str(e)}")
            
    def get_database_url(self) -> str:
        """Get the database connection URL"""
        return (f"mysql+mysqlconnector://{self.db_config['user']}:{self.db_config['password']}@"
                f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['name']}")
    
    def setup_logging(self):
        """Configure logging for the application"""
        log_file = self.logs_dir / "app.log"
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        # Create a logger for this module
        logger = logging.getLogger(__name__)
        logger.info(f"Logging initialized. Log file: {log_file}")
        
        return logger
    
    def save_to_file(self, filename: str = "config.json"):
        """Save configuration to a JSON file (excluding sensitive data)"""
        # Create a copy without sensitive information
        safe_config = {
            "api_config": self.api_config,
            "ml_config": self.ml_config,
            "testing": self.testing,
            # Include database info but mask password
            "db_config": {
                **self.db_config,
                "password": "********"  # Mask password
            }
        }
        
        # Save to file
        config_path = self.base_dir / filename
        with open(config_path, 'w') as f:
            json.dump(safe_config, f, indent=2)
            
        return config_path
    
    def get_model_path(self, symbol: str, algorithm: str) -> Path:
        """Get path for model files"""
        return self.models_dir / f"{symbol}_{algorithm}"
    
    def get_data_path(self, symbol: str) -> Path:
        """Get path for stock data files"""
        return self.data_dir / f"{symbol}.csv"

# Create a global config instance
config = config()
logger = config.setup_logging()