from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

print("🚀 Starting database connection test...")

# Update this with your actual database details
DATABASE_URL = "mysql+mysqlconnector://root:171205@Kunj@localhost:3306/stock_prediction"

print(f"🔗 Connecting to database: {DATABASE_URL}")

try:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    
    print("✅ Engine created successfully!")

    db = SessionLocal()
    print("✅ MySQL Database connection successful!")
    
    db.close()
except Exception as e:
    print("❌ MySQL Database connection failed:", e)
