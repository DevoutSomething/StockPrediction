# Stock Prediction and Trading System

A comprehensive machine learning-based system for stock prediction and automated trading using reinforcement learning algorithms. This system provides a RESTful API for stock data analysis, price predictions, and trade recommendations.

## Features

- **Stock Data API**: Fetch historical and real-time stock data
- **Price Prediction**: Machine learning-based stock price predictions
- **Reinforcement Learning Models**: DQN and PPO algorithms for automated trading
- **Portfolio Optimization**: Multi-stock portfolio optimization with risk adjustment
- **News Sentiment Analysis**: Incorporation of news sentiment into predictions
- **Database Integration**: Store predictions, stock data, and portfolio information
- **Docker Support**: Easy deployment with Docker containers

## Architecture

The system consists of the following components:

- **FastAPI Backend**: REST API for all services
- **MySQL Database**: Store stock data, predictions, and trading history
- **TensorFlow ML Models**: Deep reinforcement learning for trading decisions
- **YFinance Integration**: Real-time stock data retrieval
- **News API Integration**: News sentiment analysis

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.9+ (for local development)
- (Optional) API keys for News API

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd stock-prediction-system
   ```

2. Run the startup script:
   ```bash
   bash start.sh
   ```

   This script will:
   - Create necessary directories
   - Create a default `.env` file (edit this with your API keys)
   - Build and start Docker containers
   - Wait for the API to be ready

3. Access the API documentation:
   ```
   http://localhost:8000/docs
   ```

### Manual Setup

If you prefer to set up manually:

1. Create a `.env` file with your configuration:
   ```
   DB_USER=root
   DB_PASSWORD=your_password
   DB_HOST=db
   DB_PORT=3306
   DB_NAME=stock_prediction
   NEWS_API_KEY=your_news_api_key
   ```

2. Build and start the containers:
   ```bash
   docker-compose up -d
   ```

## API Endpoints

The system provides the following key endpoints:

- `GET /stock/{symbol}`: Get stock data for a specific symbol
- `GET /news/{symbol}`: Get latest news for a specific stock
- `POST /predict`: Predict best stocks based on investment criteria
- `GET /ml/recommend/{symbol}`: Get trading recommendation
- `GET /ml/portfolio`: Optimize a portfolio of stocks

For full API documentation, visit the `/docs` endpoint.

## Machine Learning Models

The system uses two main reinforcement learning algorithms:

### DQN (Deep Q-Network)
- Good for learning specific trading patterns
- More aggressive trading strategy

### PPO (Proximal Policy Optimization)
- More stable learning and smoother trading
- Better for long-term investment strategies and risk management

## Development

### Project Structure

```
stock-prediction-system/
├── config.py              # Configuration management
├── database/              # Database models and connection
├── main.py                # Main application entry point
├── ml_endpoints.py        # ML API endpoints
├── ml_integration.py      # Integration with ML models
├── models/                # Saved ML models
├── tests/                 # Test modules
├── tf_train.py            # ML model training
├── utils.py               # Utility functions
├── docker-compose.yml     # Docker Compose configuration
└── Dockerfile             # Docker build configuration
```

### Running Tests

```bash
# Run API tests
pytest test_utils.py

# Run ML tests
pytest test_tf_train.py
```

### Local Development

For local development without Docker:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a local MySQL database:
   ```bash
   mysql -u root -p -e "CREATE DATABASE stock_prediction;"
   ```

3. Set environment variables:
   ```bash
   export DB_HOST=localhost
   export DB_USER=root
   export DB_PASSWORD=your_password
   ```

4. Run the application:
   ```bash
   python main.py
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YFinance for providing stock data
- FastAPI for the excellent API framework
- TensorFlow for machine learning capabilities