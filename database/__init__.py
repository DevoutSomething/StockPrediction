from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Create a new metadata instance for the application
metadata = MetaData()

# Create the base class with the metadata
Base = declarative_base(metadata=metadata)

# SQLite connection string
SQLALCHEMY_DATABASE_URL = "sqlite:///./app.db"

# Create the engine
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Create sessionmaker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()