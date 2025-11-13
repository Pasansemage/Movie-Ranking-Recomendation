import requests
import json

def test_api():
    """Test the recommendation API"""
    base_url = "http://localhost:8000"
    
    # Test data
    test_request = {
        "user_id": 196,
        "candidate_movies": [1, 50, 100, 181, 258],
        "use_baseline": False
    }
    
    try:
        # Test ML model
        response = requests.post(f"{base_url}/rank", json=test_request)
        if response.status_code == 200:
            rankings = response.json()
            print("ML Model Rankings:")
            for i, movie in enumerate(rankings):
                print(f"{i+1}. {movie['title']} - Rating: {movie['predicted_rating']:.2f}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
        
        # Test baseline model
        test_request["use_baseline"] = True
        response = requests.post(f"{base_url}/rank", json=test_request)
        if response.status_code == 200:
            rankings = response.json()
            print("\nBaseline Model Rankings:")
            for i, movie in enumerate(rankings):
                print(f"{i+1}. {movie['title']} - Rating: {movie['predicted_rating']:.2f}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("API server not running. Start it with: python api.py")

if __name__ == "__main__":
    test_api()