from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import Optional, List
import pandas as pd
from pydantic import BaseModel
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os
import time
from functools import wraps
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_connection import Base, engine, SessionLocal, get_db
from backend.model_integration import ModelManager 
from backend.models import Stock, News, Prediction
from backend.tf_train import train_rl_agent, evaluate_agent, StockDataPreprocessor, get_mock_data

# Pydantic Models
class PredictionRequest(BaseModel):
    investment_amount: float
    time_frame: int  # in days
    target_return: float

class StockPrediction(BaseModel):
    symbol: str
    current_price: float
    predicted_price: float
    confidence: float
    expected_return: float

# Create a router instead of an app
router = APIRouter(tags=["Stock Data"])

# Dependency to get DB session
def create_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def retry_on_failure(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

@retry_on_failure(max_retries=3, delay=1)
def get_stock_info(symbol: str, period: str = "1mo"):
    """
    Fetch stock information with retries and improved error handling.
    
    Args:
        symbol: Stock ticker symbol
        period: Time period for historical data
        
    Returns:
        Dictionary containing stock data and information
    """
    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period=period)
        
        # Check if history is already a dict (for testing)
        if isinstance(history, dict):
            history_data = history
            is_empty = not bool(history)
        else:
            # For real data (pandas DataFrame)
            history_data = history.to_dict('list') if not history.empty else {}
            is_empty = history.empty
        
        if is_empty:
            raise HTTPException(status_code=400, detail=f"No data found for symbol {symbol}")
        
        # Get basic info but handle potential API errors
        try:
            info = ticker.info
        except Exception as e:
            print(f"Warning: Could not fetch complete info for {symbol}: {str(e)}")
            info = {"shortName": symbol, "symbol": symbol}
        
        return {
            "symbol": symbol,
            "data": history_data,
            "info": info
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error fetching data: {str(e)}")

def get_stock_history(symbol: str, period: str = "1y", interval: str = "1d"):
    """
    Get historical stock data with support for different intervals.
    
    Args:
        symbol: Stock ticker symbol
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
    Returns:
        Pandas DataFrame with historical data or None if error
    """
    # Set USE_MOCK_DATA to 1 to force using mock data during development
    if os.environ.get("USE_MOCK_DATA") == "1":
        print(f"Using mock data for {symbol}")
        # Import locally to avoid circular imports
        from backend.tf_train import get_mock_data
        return get_mock_data(symbol, period)
    
    for attempt in range(3):  # Retry up to 3 times
        try:
            data = yf.download(symbol, period=period, interval=interval, progress=False)
            
            if data.empty:
                print(f"No data available for {symbol} with period={period}, interval={interval}")
                if attempt < 2:  # Try with a longer period on failure
                    periods = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]
                    if period in periods:
                        next_period_idx = min(periods.index(period) + 1, len(periods) - 1)
                        period = periods[next_period_idx]
                        print(f"Retrying with longer period: {period}")
                        continue
                return None
            
            return data
        except Exception as e:
            if attempt < 2:
                print(f"Attempt {attempt+1} failed for {symbol}: {str(e)}. Retrying...")
                time.sleep(1 * (attempt + 1))  # Increasing backoff
            else:
                print(f"All attempts failed for {symbol}: {str(e)}")
                return None

# News fetching (you'll need a NewsAPI key)
def get_stock_news(symbol: str, api_key: str = os.getenv("NEWS_API_KEY", "your_api_key_here")):
    url = f"https://newsapi.org/v2/everything?q={symbol}+stock&apiKey={api_key}&pageSize=10&language=en"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("articles", [])
        return []
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

# Simple sentiment analysis (in a real app, use a proper NLP model)
def analyze_sentiment(text: str) -> float:
    # This is a placeholder. In a real app, use NLTK, TextBlob, or a ML model
    positive_words = ["growth", "profit", "increase", "up", "gain", "positive", "rise"]
    negative_words = ["loss", "drop", "decrease", "down", "negative", "fall", "decline"]
    
    if not text:
        return 0
    
    text = text.lower()
    pos_count = sum(1 for word in positive_words if word in text)
    neg_count = sum(1 for word in negative_words if word in text)
    
    total = pos_count + neg_count
    if total == 0:
        return 0
    
    return (pos_count - neg_count) / total  # Range: -1 to 1

# Simple stock prediction model
def predict_stock_price(symbol: str, days_ahead: int, db: Session):
    # Get historical data
    hist = get_stock_history(symbol, period="2y")
    if hist is None or len(hist) < 30:  # Need enough data
        return None
    
    # Feature engineering
    df = hist.copy()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['Close'].rolling(window=20).std()
    
    # Shift to create target variable (future price)
    df['Target'] = df['Close'].shift(-days_ahead)
    
    # Drop NaN values
    df = df.dropna()
    
    # Features and target
    features = ['Close', 'Volume', 'MA5', 'MA20', 'MA50', 'Volatility']
    X = df[features].values
    y = df['Target'].values
    
    # Split into train and test
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    
    # Train a simple model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Get current features for prediction
    current_features = df[features].iloc[-1:].values
    
    # Make prediction
    predicted_price = float(model.predict(current_features)[0])
    current_price = float(df['Close'].iloc[-1])
    
    # Calculate confidence based on model's performance
    confidence = model.score(X_train, y_train)
    
    # Get news sentiment to adjust prediction
    news_items = get_stock_news(symbol)
    sentiment_scores = []
    for news in news_items:
        sentiment = analyze_sentiment(news.get("title", "") + " " + news.get("description", ""))
        sentiment_scores.append(sentiment)
        
        # Save news to database
        db_news = News(
            title=news.get("title", ""),
            summary=news.get("description", ""),
            sentiment=sentiment,
            url=news.get("url", ""),
            symbol=symbol,
            published_at=datetime.now()
        )
        db.add(db_news)
    
    # Adjust prediction based on sentiment
    if sentiment_scores:
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        sentiment_factor = 1 + (avg_sentiment * 0.05)  # 5% adjustment
        predicted_price *= sentiment_factor
    
    # Save prediction to database
    db_prediction = Prediction(
        symbol=symbol,
        current_price=current_price,
        predicted_price=predicted_price,
        time_frame=days_ahead,
        investment_amount=0,  # Will update later
        predicted_return=0,  # Will update later
        confidence=confidence
    )
    db.add(db_prediction)
    db.commit()
    
    return {
        "symbol": symbol,
        "current_price": current_price,
        "predicted_price": predicted_price,
        "confidence": confidence
    }

# Endpoints
@router.get("/")
def read_root():
    return {"Hello": "World"}

@router.get("/stock/{symbol}")
def get_stock_data(symbol: str, period: Optional[str] = "1mo"):
    return get_stock_info(symbol, period)

@router.get("/news/{symbol}")
async def get_news_for_stock(symbol: str, db: Session = Depends(create_db_session)):
    """Get latest news for a specific stock"""
    news_items = get_stock_news(symbol)
    
    # Save to database
    for news in news_items:
        sentiment = analyze_sentiment(news.get("title", "") + " " + news.get("description", ""))
        db_news = News(
            title=news.get("title", ""),
            summary=news.get("description", ""),
            sentiment=sentiment,
            url=news.get("url", ""),
            symbol=symbol,
            published_at=datetime.now()
        )
        db.add(db_news)
    db.commit()
    
    return {"news": news_items}

@router.post("/predict")
async def predict_investment(request: PredictionRequest, db: Session = Depends(create_db_session)):
    """
    Predict best stocks based on user's investment criteria
    """
    # Top stocks to analyze - in a real app, this would be more dynamic
    top_stocks = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", "V", "PG"]
    
    predictions = []
    for symbol in top_stocks:
        try:
            prediction = predict_stock_price(symbol, request.time_frame, db)
            if prediction:
                current_price = prediction["current_price"]
                predicted_price = prediction["predicted_price"]
                confidence = prediction["confidence"]
                
                # Calculate expected return
                num_shares = request.investment_amount / current_price
                future_value = num_shares * predicted_price
                expected_return = ((future_value - request.investment_amount) / request.investment_amount) * 100
                
                # Update prediction in database
                db_prediction = db.query(Prediction).filter(
                    Prediction.symbol == symbol,
                    Prediction.time_frame == request.time_frame
                ).order_by(Prediction.created_at.desc()).first()
                
                if db_prediction:
                    db_prediction.investment_amount = request.investment_amount
                    db_prediction.predicted_return = expected_return
                    db.commit()
                
                predictions.append(StockPrediction(
                    symbol=symbol,
                    current_price=current_price,
                    predicted_price=predicted_price,
                    confidence=confidence,
                    expected_return=expected_return
                ))
        except Exception as e:
            print(f"Error predicting {symbol}: {e}")
    
    # Sort by expected return, highest first
    predictions.sort(key=lambda x: x.expected_return, reverse=True)
    
    # Filter by target return if provided
    if request.target_return > 0:
        predictions = [p for p in predictions if p.expected_return >= request.target_return]
    
    # Return top 5 predictions
    return {"predictions": predictions[:5]}

@router.get("/predict/{symbol}")
async def predict_stock(symbol: str, days_ahead: int = 7, db: Session = Depends(create_db_session)):
    """
    Predict stock price using the RL model
    """
    try:
        # Initialize preprocessor
        preprocessor = StockDataPreprocessor()
        
        # Get and preprocess data with error handling
        try:
            normalized_data, original_data = preprocessor.prepare_data(symbol, period="1y")
            if normalized_data.empty or original_data.empty:
                raise HTTPException(status_code=400, detail=f"No data available for {symbol}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error preprocessing data: {str(e)}")
        
        # Verify we have enough data
        if len(normalized_data) < 30:  # Minimum data requirement
            raise HTTPException(status_code=400, detail=f"Insufficient historical data for {symbol}")
        
        # Get current price with validation
        try:
            current_price = float(original_data['Close'].iloc[-1])
            if current_price <= 0:
                raise ValueError("Invalid current price")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error getting current price: {str(e)}")
        
        # Train the model with error handling
        try:
            agent, prep, rewards, portfolio_values = train_rl_agent(
                symbol=symbol,
                algorithm="DQN",
                episodes=10,  # Reduced episodes for faster prediction
                initial_balance=10000  # Set a reasonable initial balance
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")
        
        # Evaluate agent with error handling
        try:
            eval_results = evaluate_agent(
                agent=agent,
                symbol=symbol,
                algorithm="DQN",
                preprocessor=prep,
                start_date=None,  # Will use most recent data
                initial_balance=10000  # Set same initial balance as training
            )
            
            if not eval_results:
                raise HTTPException(status_code=500, detail="Evaluation failed to return results")
            
            predicted_return = eval_results.get('agent_return', 0)
            sharpe_ratio = eval_results.get('sharpe_ratio', 0)
            
            # Calculate predicted price with safety checks
            predicted_price = current_price * (1 + (predicted_return/100))
            if predicted_price <= 0:
                raise ValueError("Invalid predicted price calculated")
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during evaluation: {str(e)}")
        
        # Save prediction to database
        try:
            db_prediction = Prediction(
                symbol=symbol,
                current_price=current_price,
                predicted_price=predicted_price,
                confidence=sharpe_ratio,
                time_frame=days_ahead,
                created_at=datetime.now()
            )
            db.add(db_prediction)
            db.commit()
        except Exception as e:
            print(f"Warning: Could not save prediction to database: {str(e)}")
            # Don't fail the request if database save fails
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "confidence": sharpe_ratio,
            "predicted_return": predicted_return,
            "time_frame_days": days_ahead
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

class PortfolioRequest(BaseModel):
    symbols: Optional[List[str]] = None
    investment_amount: float
    risk_level: float  # On a scale of 0-10

@router.post("/optimize-portfolio/")
async def optimize_portfolio(
    request: PortfolioRequest,
    db: Session = Depends(create_db_session)
):
    """
    Optimize portfolio based on user criteria
    """
    try:
        # Default symbols if none provided
        symbols = request.symbols or ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        
        # Initialize preprocessor
        preprocessor = StockDataPreprocessor()
        
        portfolio_allocations = {}
        total_predicted_return = 0
        
        # Analyze each stock
        for symbol in symbols:
            # Get prediction for each stock
            prediction = await predict_stock(symbol)
            
            # Calculate allocation based on predicted return and risk level
            weight = max(0.1, min(0.4, prediction['predicted_return'] / 100))  # Cap between 10-40%
            
            # Adjust weight based on risk level (0-1)
            risk_adjusted_weight = weight * (request.risk_level / 10)
            
            portfolio_allocations[symbol] = {
                "allocation": risk_adjusted_weight,
                "amount": request.investment_amount * risk_adjusted_weight,
                "predicted_return": prediction['predicted_return']
            }
            
            total_predicted_return += prediction['predicted_return'] * risk_adjusted_weight
        
        # Normalize allocations to sum to 1
        total_weight = sum(stock['allocation'] for stock in portfolio_allocations.values())
        for symbol in portfolio_allocations:
            portfolio_allocations[symbol]['allocation'] /= total_weight
            portfolio_allocations[symbol]['amount'] = request.investment_amount * portfolio_allocations[symbol]['allocation']
        
        return {
            "portfolio_allocations": portfolio_allocations,
            "total_investment": request.investment_amount,
            "expected_return": total_predicted_return,
            "risk_level": request.risk_level
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portfolio optimization error: {str(e)}")

# Add health check endpoint
@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

# Export the router instead of the app