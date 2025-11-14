import requests
import json

user_id = 685

def test_api():
    """Test the movie recommendation API with XGBoost model"""
    base_url = "http://localhost:8000"
    
    print("Testing Movie Recommendation API with XGBoost model...")
    print(f"User {user_id} movie recommendations:")
    print("="*50)
    
    # Test data (using user with movies they've rated)
    test_request = {
        "user_id": user_id,
        "candidate_movies": [56, 593, 158, 111, 582, 1400, 953, 995, 386],
        "use_baseline": False
    }
    
    try:
        # Test ML model
        response = requests.post(f"{base_url}/rank", json=test_request)
        if response.status_code == 200:
            rankings = response.json()
            print("XGBoost Model Rankings:")
            for i, movie in enumerate(rankings):
                method = movie.get('method', 'unknown')
                print(f"{i+1}. {movie['title']} - Rating: {movie['predicted_rating']:.2f} ({method})")
        else:
            print(f"Error: {response.status_code} - {response.text}")
        
        # Test baseline model
        test_request["use_baseline"] = True
        response = requests.post(f"{base_url}/rank", json=test_request)
        if response.status_code == 200:
            rankings = response.json()
            print("\nBaseline Model Rankings:")
            for i, movie in enumerate(rankings):
                method = movie.get('method', 'baseline')
                print(f"{i+1}. {movie['title']} - Rating: {movie['predicted_rating']:.2f} ({method})")
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("API server not running. Start it with: python api.py")
        
    # Additional test cases
    print("\n" + "="*50)
    print("Testing edge cases...")
    
    try:
        # Test invalid user
        invalid_user_request = {
            "user_id": 99999,
            "candidate_movies": [1, 2, 3],
            "use_baseline": False
        }
        response = requests.post(f"{base_url}/rank", json=invalid_user_request)
        print(f"Invalid user test: {response.status_code} - {response.json().get('detail', 'No detail')}")
        
        # Test invalid movies
        invalid_movies_request = {
            "user_id": user_id,
            "candidate_movies": [99999, 88888],
            "use_baseline": False
        }
        response = requests.post(f"{base_url}/rank", json=invalid_movies_request)
        print(f"Invalid movies test: {response.status_code} - {response.json().get('detail', 'No detail')}")
        
    except requests.exceptions.ConnectionError:
        print("API server not running for edge case tests")

if __name__ == "__main__":
    test_api()