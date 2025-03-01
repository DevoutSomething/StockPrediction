from sqlalchemy import Column, Integer, String, Float, DateTime
from database.db_connection import Base
import datetime

class Stock(Base):
    __tablename__ = "stocks"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), unique=True, index=True)
    name = Column(String(100))
    price = Column(Float)
    date = Column(DateTime, default=datetime.datetime.utcnow)

class News(Base):
    __tablename__ = "news"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255))
    summary = Column(String(500))
    url = Column(String(255), unique=True)
    published_at = Column(DateTime, default=datetime.datetime.utcnow)

