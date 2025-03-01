from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import logging

# Get database configuration from environment variables with defaults
DB_USER = os.environ.get("DB_USER", "root")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "171205@Kunj")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "3306")
DB_NAME = os.environ.get("DB_NAME", "stock_prediction")

# For testing purposes
# Find code like this
if os.environ.get("TESTING") == "1":
    DB_NAME = "stock_prediction_test"

# Force SQLite mode for development without MySQL
USE_SQLITE = True
DATABASE_URL = "sqlite:///./app.db"

print(f"Using SQLite database: {DATABASE_URL}")

# Create engine with SQLite-compatible settings
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()