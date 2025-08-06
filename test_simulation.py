#!/usr/bin/env python3
"""
Test script to verify the simulation endpoint works correctly
"""

import requests
import json

def test_simulation():
    url = "http://localhost:8000/api/v1/simulate"
    
    payload = {
        "news_inputs": [
            {
                "content": "Apple reports record quarterly earnings with strong iPhone sales",
                "source_type": "headline"
            }
        ],
        "target_assets": ["AAPL"],
        "simulation_method": "hybrid_qml",
        "time_horizon_days": 7,
        "num_scenarios": 1000
    }
    
    try:
        print("Testing simulation endpoint...")
        response = requests.post(url, json=payload)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Simulation successful!")
            print(f"Request ID: {data.get('request_id')}")
            print(f"Market Predictions Count: {len(data.get('market_predictions', []))}")
            
            # Check if MarketPrediction has all required fields
            if data.get('market_predictions'):
                prediction = data['market_predictions'][0]
                required_fields = ['current_price', 'predicted_scenarios', 'confidence_intervals', 'prediction_method']
                
                print("\nChecking MarketPrediction fields:")
                for field in required_fields:
                    if field in prediction:
                        print(f"✅ {field}: {type(prediction[field])}")
                    else:
                        print(f"❌ Missing field: {field}")
                        
        else:
            print(f"❌ Simulation failed with status {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to backend. Make sure it's running on localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_simulation()
