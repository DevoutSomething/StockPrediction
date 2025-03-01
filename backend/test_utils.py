import pytest
from fastapi.testclient import TestClient
from backend.utils import app
import yfinance as yf
from unittest.mock import patch

client = TestClient(app)


# Mock stock data for testing
MOCK_STOCK_DATA = {
    "symbol": "AAPL",
    "data": {
        "Open": [150.0],
        "High": [155.0],
        "Low": [149.0],
        "Close": [153.0],
        "Volume": [1000000]
    },
    "info": {
        "shortName": "Apple Inc.",
        "sector": "Technology"
    }
}

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}

@pytest.fixture
def mock_yfinance():
    with patch('yfinance.Ticker') as mock_ticker:
        # Configure the mock to return our test data
        mock_instance = mock_ticker.return_value
        mock_instance.history.return_value = MOCK_STOCK_DATA["data"]
        mock_instance.info = MOCK_STOCK_DATA["info"]
        yield mock_ticker

def test_get_stock_data(mock_yfinance):
    # Test valid stock symbol
    response = client.get("/stock/AAPL")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "AAPL"
    assert "data" in data
    assert "info" in data

    # Test invalid stock symbol
    response = client.get("/stock/INVALID_SYMBOL")
    assert response.status_code == 400

    # Test different period
    response = client.get("/stock/AAPL?period=5d")
    assert response.status_code == 200
    assert response.json()["symbol"] == "AAPL" 