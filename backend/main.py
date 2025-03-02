import uvicorn
from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import os
from backend.utils import (
    get_stock_info, 
    get_stock_news,
    predict_stock_price,
    router as utils_router,
    PortfolioRequest  # Add this import
)
from sqlalchemy.orm import Session
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Import configuration
from backend.config import config, logger

# Import database models and connection
from database.db_connection import get_db, engine
from database import models

# Import utils - change how we import utils
from backend.utils import (
    get_stock_info, 
    get_stock_news,
    predict_stock_price,
    router as utils_router  # Import the router instead of the app
)

# Import ML router
try:
    from backend.ml_endpoints import router as ml_router
    HAS_ML = True
except ImportError:
    print("Machine learning module not available")
    HAS_ML = False

# Create main FastAPI application
app = FastAPI(
    title="Stock Prediction and Trading API",
    description="An API for stock prediction and algorithmic trading using reinforcement learning",
    version="1.0.0",
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Add error handling middleware
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again later."}
    )

# Include routers
app.include_router(utils_router)  # Use the router from utils

if HAS_ML:
    app.include_router(ml_router)

# Mount static files for frontend if the directory exists
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="static")

# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "Stock Prediction API is running",
        "docs_url": "/docs",
        "ml_available": HAS_ML
    }

# Health check endpoint
@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    try:
        # Check database connection
        db.execute("SELECT 1")
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "database": db_status,
        "environment": "testing" if config.testing else "production"
    }

# Check API keys endpoint (admin only in a real app)
@app.get("/check-api-keys")
def check_api_keys():
    # This would normally be protected by authentication
    return {
        "news_api": bool(config.api_keys["news_api"]),
        "alpha_vantage": bool(config.api_keys["alpha_vantage"])
    }

if __name__ == "__main__":
    # Create database tables if they don't exist
    from database.db_connection import Base
    Base.metadata.create_all(bind=engine)
    
    # Run the application
    uvicorn.run(
        "main:app", 
        host=config.api_config["host"],
        port=config.api_config["port"],
        reload=config.api_config["reload"]
    )