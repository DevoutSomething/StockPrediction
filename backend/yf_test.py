import yfinance as yf
print(yf.__version__)

# Test a simple fetch
aapl = yf.Ticker("AAPL")
print(aapl.info)