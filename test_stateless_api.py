#!/usr/bin/env python3
"""
Simple test script to verify the stateless API functionality.
"""

import requests


def test_stateless_api():
    """Test the stateless prediction API."""
    
    base_url = "http://localhost:5000"
    
    # Test health endpoint
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            if health_data.get("status") == "ok":
                print("Health check passed: status ok")
            else:
                print(f"Health check failed: unexpected body {health_data}")
                return False
        else:
            print(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Health check error: {e}")
        return False
    
    # Test prediction endpoint
    print("\nTesting stateless prediction endpoint...")
    
    # Create test payload
    test_payload = {
        "x_observed": [
            [0.5, -1.2, 0.8],
            [1.0, 0.2, -0.5],
            [-0.3, 1.5, 0.1],
            [0.8, -0.7, 1.3],
            [1.2, 0.9, -0.4],
            [-0.5, 0.3, 0.9]
        ],
        "y_observed": [2.1, 1.8, -0.5, 2.3, 3.1, 0.8],
        "x_predict": [
            [0.7, -1.1, 0.2],
            [1.5, 0.6, -0.9]
        ],
        "t": "The output y is a linear combination of the input features with some noise.",
        "n": 6,
        "k": 2,
        "d": 3
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=test_payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == 'success':
                predictions = result['predictions']
                print(f"Prediction successful: {predictions}")
                print(f"   Predicted {len(predictions)} values as expected")
                return True
            else:
                print(f"Prediction failed: {result}")
                return False
        else:
            print(f"Prediction request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Prediction error: {e}")
        return False


def test_error_handling():
    """Test API error handling with invalid payloads."""
    
    base_url = "http://localhost:5000"
    
    print("\nTesting error handling...")
    
    # Test missing field
    invalid_payload = {
        "x_observed": [[1, 2, 3]],
        "y_observed": [1.5],
        # Missing x_predict, t, n, k, d
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=invalid_payload,
            timeout=5
        )
        
        if response.status_code == 400:
            error_data = response.json()
            print(f"Error handling works: {error_data}")
            return True
        else:
            print(f"Expected 400 error, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Error handling test failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing Stateless API")
    print("=" * 40)
    
    # Run tests
    health_ok = test_stateless_api()
    error_ok = test_error_handling()
    
    print("\nTest Results:")
    print(f"   Health & Prediction: {'PASS' if health_ok else 'FAIL'}")
    print(f"   Error Handling: {'PASS' if error_ok else 'FAIL'}")
    
    if health_ok and error_ok:
        print("\nAll tests passed! The stateless API is working correctly.")
    else:
        print("\nSome tests failed. Please check the API implementation.")
