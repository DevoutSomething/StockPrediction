from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from ml_integration import get_trading_recommendation, model_manager
from database.db_connection import get_db
import yfinance as yf

# backend/ml_endpoints.py
from fastapi import APIRouter

router = APIRouter()


# Pydantic models for request/response
class MLPredictionRequest(BaseModel):
    symbol: str
    algorithm: str = "PPO"  # Default to PPO
    investment_amount: float = 10000.0
    days_ahead: int = 30

class MLModelStatus(BaseModel):
    symbol: str
    algorithm: str
    is_trained: bool
    last_updated: Optional[str] = None

class TradeRecommendation(BaseModel):
    symbol: str
    current_price: float
    recommendation: str  # BUY, SELL, HOLD
    confidence: float
    expected_return_pct: float
    potential_value: float
    algorithm: str
    units_to_trade: float
    prediction_date: str

# Create router
router = APIRouter(prefix="/ml", tags=["Machine Learning"])

@router.get("/status/{symbol}")
async def get_model_status(symbol: str):
    """Check if models are trained for a specific symbol"""
    dqn_trained = model_manager.has_trained_model(symbol, "DQN")
    ppo_trained = model_manager.has_trained_model(symbol, "PPO")
    
    return {
        "symbol": symbol,
        "models": {
            "DQN": {"trained": dqn_trained},
            "PPO": {"trained": ppo_trained}
        }
    }

@router.post("/train/{symbol}")
async def train_model(symbol: str, algorithm: str = "PPO", force: bool = False):
    """Train a model for a specific symbol"""
    try:
        # Verify the symbol is valid
        ticker = yf.Ticker(symbol)
        info = ticker.info
        if not info or "regularMarketPrice" not in info:
            raise HTTPException(status_code=400, detail=f"Invalid symbol: {symbol}")
        
        # Train the model
        result = model_manager.get_or_train_model(symbol, algorithm, force_retrain=force)
        
        return {
            "symbol": symbol,
            "algorithm": algorithm,
            "status": "completed" if result else "failed",
            "newly_trained": result.get("newly_trained", True)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@router.get("/recommend/{symbol}")
async def get_recommendation(
    symbol: str, 
    algorithm: str = "PPO", 
    investment_amount: float = 10000.0,
    db: Session = Depends(get_db)
):
    """Get trading recommendation for a stock"""
    try:
        recommendation = get_trading_recommendation(
            symbol=symbol,
            algorithm=algorithm,
            investment_amount=investment_amount
        )
        
        # Save recommendation to database if needed
        # This could be implemented in your database models
        
        return recommendation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendation: {str(e)}")

@router.get("/portfolio")
async def optimize_portfolio(
    symbols: List[str], 
    investment_amount: float = 10000.0,
    risk_factor: float = 0.5,
    db: Session = Depends(get_db)
):
    """Optimize a portfolio of stocks"""
    if not symbols:
        raise HTTPException(status_code=400, detail="No symbols provided")
    
    if len(symbols) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 symbols allowed")
    
    try:
        from tf_train import multi_stock_portfolio_optimization
        
        # Run portfolio optimization
        portfolio_summary, agents, preprocessors = multi_stock_portfolio_optimization(
            symbols=symbols,
            investment_amount=investment_amount,
            algorithm="PPO",
            risk_factor=risk_factor
        )
        
        # Format and return the results
        return {
            "portfolio": {
                "symbols": symbols,
                "investment_amount": investment_amount,
                "risk_factor": risk_factor,
                "expected_return": portfolio_summary["expected_return"],
                "expected_sharpe": portfolio_summary["expected_sharpe"],
                "allocations": {
                    symbol: {
                        "percentage": allocation * 100,
                        "amount": investment_amount * allocation
                    } for symbol, allocation in portfolio_summary["allocations"].items()
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portfolio optimization failed: {str(e)}")