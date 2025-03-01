import requests
import json

BASE_URL = "http://localhost:8001"

def test_endpoint(endpoint, method="GET", data=None):
    """Test an API endpoint and print the result"""
    url = f"{BASE_URL}{endpoint}"
    print(f"\n🔍 Testing {method} {url}")
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        else:
            print(f"❌ Unsupported method: {method}")
            return
        
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Success!")
            try:
                print(json.dumps(response.json(), indent=2))
            except:
                print(response.text[:100] + "...")
        else:
            print(f"❌ Error: {response.text}")
    except Exception as e:
        print(f"❌ Exception: {str(e)}")

# Test all endpoints
print("🚀 Testing Stock Prediction API")

# Basic endpoints
test_endpoint("/")
test_endpoint("/health")

# Stock data
test_endpoint("/stock/AAPL")
test_endpoint("/stock/MSFT")

# News
test_endpoint("/news/AAPL")

# Predictions
test_endpoint("/predict/AAPL")
test_endpoint("/predict", method="POST", data={
    "investment_amount": 10000,
    "time_frame": 30,
    "target_return": 10
})

# Portfolio optimization
test_endpoint("/optimize-portfolio/", method="POST", data={
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "investment_amount": 10000,
    "risk_level": 5
})

print("\n🏁 Testing complete!") 