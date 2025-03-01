# backend/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from datetime import datetime
from database import Base


class Stock(Base):
    __tablename__ = 'stocks'
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), unique=True, index=True)
    name = Column(String(100))
    price = Column(Float)
    date = Column(DateTime, default=datetime.utcnow)

class News(Base):
    __tablename__ = "news"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255))
    summary = Column(Text)
    sentiment = Column(Float)
    url = Column(String(255))
    symbol = Column(String(10), index=True)
    published_at = Column(DateTime, default=datetime.utcnow)

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True)
    current_price = Column(Float)
    predicted_price = Column(Float)
    time_frame = Column(Integer)
    investment_amount = Column(Float)
    predicted_return = Column(Float)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)