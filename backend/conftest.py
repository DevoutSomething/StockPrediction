import os
import sys
import pytest
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.append(str(backend_dir.parent))

# Set environment variables for testing
os.environ["TESTING"] = "1"
os.environ["DB_URL"] = "mysql+mysqlconnector://root:171205@Kunj@localhost:3306/stock_prediction"

@pytest.fixture(autouse=True)
def setup_test_env():
    """Setup test environment variables and database"""
    from database.db_connection import Base, engine
    
    # Create test database tables
    Base.metadata.create_all(bind=engine)
    
    yield
    
    # Clean up test database
    Base.metadata.drop_all(bind=engine) 