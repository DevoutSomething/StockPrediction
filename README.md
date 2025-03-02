# Enhanced Stock Risk Prediction System

A comprehensive TensorFlow-based deep learning system for predicting stock risk levels and optimizing investment portfolios based on user preferences and real-time market data.

## New Features

- **Real-Time Data Fetching**: Toggle option to fetch the latest stock data from Yahoo Finance
- **Portfolio Optimization**: Automatically select the best stocks for a custom portfolio based on risk tolerance
- **Large-Scale Support**: Optimized to handle 3000+ companies in your dataset
- **Multiple Scenarios**: Run and compare different investment scenarios
- **Configuration Management**: Flexible configuration system for easy customization

## System Requirements

- Python 3.8+
- TensorFlow 2.6+
- pandas, numpy, scikit-learn
- Flask
- Matplotlib, Seaborn
- SHAP
- yfinance (for real-time data)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/stock-risk-prediction.git
cd stock-risk-prediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Setting Up Configuration

```bash
# Initialize default configuration
python main.py config

# Enable real-time data fetching
python main.py config --toggle-real-time on

# Set maximum portfolio size
python main.py config --set portfolio max_portfolio_size 15
```

### Training the Model

```bash
# Train with default settings
python main.py train --data data/stock_data.csv

# Train with custom settings
python main.py train --data data/stock_data.csv --stocks AAPL MSFT GOOGL --epochs 100 --batch-size 64 --output custom_model
```

### Fetching the Latest Data

```bash
# Update all stock data with the latest from Yahoo Finance
python main.py update-data --data data/stock_data.csv

# Update specific symbols
python main.py update-data --data data/stock_data.csv --symbols AAPL TSLA MSFT
```

### Optimizing a Portfolio

```bash
# Create an optimized portfolio based on model predictions
python main.py optimize --model models/stock_prediction_model.h5 --data data/stock_data.csv --scalers models/scalers --investment 10000 --return 15 --horizon 365 --risk Medium --real-time
```

### Starting the API Server

```bash
# Start the API server
python main.py serve --config config.json --real-time
```

## API Usage

### Making a Prediction and Portfolio Optimization

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "investment_amount": 10000,
    "expected_return": 15,
    "time_horizon": 365,
    "symbols": ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT"],
    "use_real_time": true
  }'
```

### Toggling Real-Time Data

```bash
curl -X POST http://localhost:5000/toggle-real-time \
  -H "Content-Type: application/json" \
  -d '{
    "use_real_time": true
  }'
```

### Running Multiple Scenarios

```bash
curl -X POST http://localhost:5000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "scenarios": [
      {
        "investment_amount": 10000,
        "expected_return": 15,
        "time_horizon": 365,
        "risk_tolerance": "Low"
      },
      {
        "investment_amount": 10000,
        "expected_return": 15,
        "time_horizon": 365,
        "risk_tolerance": "High"
      }
    ],
    "symbols": ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT"],
    "use_real_time": true
  }'
```

### Example Response

```json
{
  "predictions": {
    "stock_recommendations": {
      "AAPL": {
        "recommendation": "Buy",
        "confidence": 0.82
      },
      "GOOGL": {
        "recommendation": "Hold",
        "confidence": 0.55
      },
      ...
    }
  },
  "optimized_portfolio": {
    "stocks": [
      {
        "symbol": "AAPL",
        "recommendation": "Buy",
        "confidence": 0.82,
        "risk": 0.18,
        "allocation_percentage": 28.5,
        "investment_amount": 2850.0
      },
      ...
    ],
    "risk_level": "Medium",
    "total_investment": 10000,
    "expected_return": 15,
    "time_horizon": 365,
    "portfolio_size": 5,
    "average_confidence": 0.72,
    "average_risk": 0.28
  }
}
```

## Project Structure

```
stock-prediction-system/
├── data/
│   ├── stock_data.csv
│   └── cache/                 # Cache for real-time data
├── models/
│   ├── stock_prediction_model.h5
│   ├── training_results.json
│   ├── scalers/
│   └── visualizations/
├── stock_data_preprocessing.py  # Data preprocessing module
├── model_architecture.py        # LSTM model architecture
├── training_pipeline.py         # Training pipeline
├── api_service.py               # Flask API service
├── real_time_data.py            # Real-time data fetching
├── portfolio_optimizer.py       # Portfolio optimization
├── config.py                    # Configuration management
├── main.py                      # Main script
├── config.json                  # System configuration
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Configuration Options

The system uses a JSON configuration file with the following sections:

### Data Settings

```json
"data": {
  "use_real_time": true,
  "cache_dir": "data/cache",
  "cache_expiry_hours": 24,
  "historical_data_path": "data/stock_data.csv"
}
```

### Model Settings

```json
"model": {
  "model_path": "models/stock_prediction_model.h5",
  "scalers_path": "models/scalers",
  "sequence_length": 30,
  "batch_size": 32,
  "epochs": 50
}
```

### Portfolio Settings

```json
"portfolio": {
  "default_risk_tolerance": "Medium",
  "max_portfolio_size": 10
}
```

### Stock Settings

```json
"stocks": {
  "default_symbols": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
  "max_stocks_to_analyze": 3000
}
```

## Customization

### Adding New Features

To add new technical indicators:

1. Modify the `engineer_features` method in `api_service.py` or `stock_data_preprocessing.py`
2. Ensure the new features are properly normalized

### Supporting More Stocks

The system can handle up to 3000 stocks by default, but this can be adjusted:

```bash
python main.py config --set stocks max_stocks_to_analyze 5000
```

### Risk Tolerance Levels

The system supports three risk tolerance levels:

- **Low**: Conservative investments with lower risk and potentially lower returns
- **Medium**: Balanced approach with moderate risk and returns
- **High**: Aggressive investments with higher risk and potentially higher returns

## Troubleshooting

### Common Issues

- **Real-Time Data Errors**: If Yahoo Finance API is unavailable, disable real-time with `--no-real-time`
- **Memory Issues**: When processing 3000+ stocks, increase your system's available memory
- **Cache Issues**: Clear the cache directory (`data/cache`) if you encounter stale data problems

## License

[MIT License](LICENSE)

## Acknowledgements

- TensorFlow and Keras for the deep learning framework
- pandas and numpy for data processing
- Flask for the API server
- SHAP for model explainability
- Yahoo Finance for real-time market data