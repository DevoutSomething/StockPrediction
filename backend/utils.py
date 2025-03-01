from fastapi import FastAPI, HTTPException, Depends
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
from database.db_connection import Base, engine, SessionLocal, get_db


# Database models
class Stock(Base):
    __tablename__ = "stocks"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, unique=True, index=True)
    name = Column(String)
    price = Column(Float)
    date = Column(DateTime, default=datetime.utcnow)

class News(Base):
    __tablename__ = "news"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    summary = Column(Text)
    sentiment = Column(Float)  # -1 to 1 sentiment score
    url = Column(String)
    symbol = Column(String, index=True)  # Associated stock symbol
    published_at = Column(DateTime, default=datetime.utcnow)

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    current_price = Column(Float)
    predicted_price = Column(Float)
    time_frame = Column(Integer)  # Time frame in days
    investment_amount = Column(Float)
    predicted_return = Column(Float)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

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

# FastAPI app
app = FastAPI(title="Stock Prediction API")

# Dependency to get DB session
def get_db():
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
    ticker = yf.Ticker(symbol)
    try:
        history = ticker.history(period=period)
        if history.empty:
            raise HTTPException(status_code=400, detail="No data found for the symbol")
        
        info = ticker.info
        return {
            "symbol": symbol,
            "data": history.to_dict('list'),
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Stock data fetching
def get_stock_history(symbol: str, period: str = "1y"):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        if hist.empty:
            return None
        return hist
    except Exception as e:
        print(f"Error fetching stock data: {e}")
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
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/stock/{symbol}")
def get_stock_data(symbol: str, period: Optional[str] = "1mo"):
    return get_stock_info(symbol, period)

@app.get("/news/{symbol}")
async def get_news_for_stock(symbol: str, db: Session = Depends(get_db)):
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

@app.post("/predict")
async def predict_investment(request: PredictionRequest, db: Session = Depends(get_db)):
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)