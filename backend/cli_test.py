import argparse
from sqlalchemy.orm import Session
from utils import predict_stock_price, SessionLocal  # Adjust the import if your main file is named differently

def main():
    parser = argparse.ArgumentParser(description='Test stock prediction for a given ticker.')
    parser.add_argument('ticker', type=str, help='Ticker symbol to test (e.g., AAPL)')
    parser.add_argument('--days', type=int, default=30, help='Prediction time frame in days')
    args = parser.parse_args()


    # Create a new DB session
    db: Session = SessionLocal()
    try:
        result = predict_stock_price(args.ticker, args.days, db)
        if result is None:
            print(f"Prediction could not be made for ticker {args.ticker}. Check if sufficient historical data is available.")
        else:
            print(f"Prediction for {args.ticker}:")
            print(f"  Current Price: {result['current_price']}")
            print(f"  Predicted Price: {result['predicted_price']}")
            print(f"  Model Confidence: {result['confidence']}")
    finally:
        db.close()

if __name__ == '__main__':
    main()
