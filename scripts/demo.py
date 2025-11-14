import pandas as pd
import joblib
import os
import sys

# Get project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.preprocessor import DataPreprocessor
from src.model import RecommendationModel, BaselineModel
from src.ranker import MovieRanker
from config.settings import *

def load_models():
    """Load pre-trained models or train new ones if not found"""
    print("Loading data and models...")
    
    # Load data with absolute path
    data_path = os.path.join(project_root, COMBINED_DATA_PATH)
    df = pd.read_csv(data_path)
    
    try:
        # Load trained models with absolute paths
        preprocessor_path = os.path.join(project_root, 'models/preprocessor.pkl')
        ml_model_path = os.path.join(project_root, 'models/ml_model.pkl')
        baseline_model_path = os.path.join(project_root, 'models/baseline_model.pkl')
        
        preprocessor = DataPreprocessor().load(preprocessor_path)
        ml_model = RecommendationModel('rf').load(ml_model_path)
        baseline_model = joblib.load(baseline_model_path)
        
        print("✓ Pre-trained models loaded successfully!")
        
    except FileNotFoundError:
        print("Pre-trained models not found. Please run 'python scripts/train_model.py' first.")
        return None, None
    
    # Create ranker
    ranker = MovieRanker(ml_model, preprocessor, baseline_model)
    ranker.set_movie_data(df)
    
    return ranker, df

def predict_all_movies_for_user(ranker, df, user_id):
    """Predict ratings for all movies for a given user"""
    # Get all unique movies
    all_movies = df['item_id'].unique().tolist()
    
    # Get user's actual ratings for comparison
    user_ratings = df[df['user_id'] == user_id]
    actual_ratings = dict(zip(user_ratings['item_id'], user_ratings['rating']))
    
    print(f"\nPredicting ratings for all {len(all_movies)} movies for User {user_id}...")
    
    # Get predictions for all movies
    predictions = ranker.rank_movies(user_id, all_movies, df, use_baseline=False)
    
    # Sort by predicted rating (highest first)
    predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
    
    print(f"\nTop 20 Movie Recommendations for User {user_id}:")
    print("=" * 120)
    print(f"{'Rank':<5} {'Movie ID':<8} {'Predicted':<9} {'Actual':<7} {'Method':<12} {'Title':<25} {'Explanation':<35}")
    print("=" * 120)
    
    for i, movie in enumerate(predictions[:20], 1):
        movie_id = movie['movie_id']
        predicted = movie['predicted_rating']
        actual = actual_ratings.get(movie_id, 'N/A')
        method = movie.get('method', 'ml_model')[:11]
        title = movie['title'][:24]
        explanation = movie.get('explanation', 'No explanation')[:34]
        
        print(f"{i:<5} {movie_id:<8} {predicted:<9.2f} {actual:<7} {method:<12} {title:<25} {explanation:<35}")
    
    # Show some statistics
    rated_movies = [p for p in predictions if p['movie_id'] in actual_ratings]
    if rated_movies:
        print(f"\nUser {user_id} Statistics:")
        print(f"• Movies already rated: {len(actual_ratings)}")
        print(f"• Movies not yet rated: {len(all_movies) - len(actual_ratings)}")
        print(f"• Average actual rating: {sum(actual_ratings.values()) / len(actual_ratings):.2f}")
        print(f"• Predicted rating range: {predictions[-1]['predicted_rating']:.2f} - {predictions[0]['predicted_rating']:.2f}")
        
        # Show accuracy on rated movies
        errors = []
        for movie in rated_movies:
            actual = actual_ratings[movie['movie_id']]
            predicted = movie['predicted_rating']
            errors.append(abs(actual - predicted))
        
        if errors:
            mae = sum(errors) / len(errors)
            print(f"• Mean Absolute Error on rated movies: {mae:.3f}")
    
    return predictions

def interactive_demo():
    """Interactive demo for movie rating predictions"""
    print("🎬 Movie Recommendation System - Interactive Demo")
    print("=" * 60)
    
    # Load models once
    ranker, df = load_models()
    if ranker is None:
        return
    
    # Get available users
    available_users = sorted(df['user_id'].unique())
    print(f"Available users: {min(available_users)} - {max(available_users)} ({len(available_users)} total users)")
    
    while True:
        print("\n" + "=" * 60)
        try:
            user_input = input("Enter User ID (or 'quit' to exit): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Thanks for using the Movie Recommendation System!")
                break
            
            user_id = int(user_input)
            
            if user_id not in available_users:
                print(f"❌ User {user_id} not found. Available range: {min(available_users)}-{max(available_users)}")
                continue
            
            # Predict all movies for this user
            predictions = predict_all_movies_for_user(ranker, df, user_id)
            
            # Ask if user wants to see more details
            while True:
                detail_input = input("\nShow more details? (top50/bottom20/rated/search/new): ").strip().lower()
                
                if detail_input == 'top50':
                    print(f"\nTop 50 Recommendations for User {user_id}:")
                    for i, movie in enumerate(predictions[:50], 1):
                        print(f"{i:2d}. {movie['title']} ({movie.get('year', 'N/A')}) - {movie['predicted_rating']:.2f}")
                
                elif detail_input == 'bottom20':
                    print(f"\nLowest 20 Predicted Ratings for User {user_id}:")
                    for i, movie in enumerate(predictions[-20:], len(predictions)-19):
                        print(f"{i:2d}. {movie['title']} ({movie.get('year', 'N/A')}) - {movie['predicted_rating']:.2f}")
                
                elif detail_input == 'rated':
                    # Show all movies user has actually rated with actual vs predicted
                    user_ratings = df[df['user_id'] == user_id]
                    actual_ratings = dict(zip(user_ratings['item_id'], user_ratings['rating']))
                    
                    # Get predictions for rated movies only
                    rated_predictions = [p for p in predictions if p['movie_id'] in actual_ratings]
                    
                    # Sort by actual rating (highest first)
                    rated_predictions.sort(key=lambda x: actual_ratings[x['movie_id']], reverse=True)
                    
                    print(f"\nAll {len(rated_predictions)} Movies User {user_id} Has Rated (Actual vs Predicted):")
                    print("=" * 120)
                    print(f"{'Rank':<5} {'Movie ID':<8} {'Actual':<7} {'Predicted':<9} {'Error':<6} {'Method':<12} {'Title':<25} {'Explanation':<30}")
                    print("=" * 120)
                    
                    total_error = 0
                    for i, movie in enumerate(rated_predictions, 1):
                        movie_id = movie['movie_id']
                        actual = actual_ratings[movie_id]
                        predicted = movie['predicted_rating']
                        error = abs(actual - predicted)
                        total_error += error
                        method = movie.get('method', 'ml_model')[:11]
                        title = movie['title'][:24]
                        explanation = movie.get('explanation', 'No explanation')[:29]
                        
                        print(f"{i:<5} {movie_id:<8} {actual:<7} {predicted:<9.2f} {error:<6.2f} {method:<12} {title:<25} {explanation:<30}")
                    
                    # Show accuracy statistics
                    mae = total_error / len(rated_predictions)
                    rmse = (sum([(actual_ratings[p['movie_id']] - p['predicted_rating'])**2 for p in rated_predictions]) / len(rated_predictions))**0.5
                    
                    print(f"\nAccuracy Statistics:")
                    print(f"• Mean Absolute Error (MAE): {mae:.3f}")
                    print(f"• Root Mean Square Error (RMSE): {rmse:.3f}")
                    print(f"• Best prediction (lowest error): {min(rated_predictions, key=lambda x: abs(actual_ratings[x['movie_id']] - x['predicted_rating']))['title']}")
                    print(f"• Worst prediction (highest error): {max(rated_predictions, key=lambda x: abs(actual_ratings[x['movie_id']] - x['predicted_rating']))['title']}")
                
                elif detail_input == 'search':
                    search_term = input("Enter movie title to search: ").strip().lower()
                    matches = [p for p in predictions if search_term in p['title'].lower()]
                    if matches:
                        print(f"\nSearch results for '{search_term}':")
                        user_ratings = df[df['user_id'] == user_id]
                        actual_ratings = dict(zip(user_ratings['item_id'], user_ratings['rating']))
                        
                        for movie in matches[:10]:
                            rank = predictions.index(movie) + 1
                            actual = actual_ratings.get(movie['movie_id'], 'N/A')
                            explanation = movie.get('explanation', 'No explanation')
                            print(f"Rank {rank}: {movie['title']} - Predicted: {movie['predicted_rating']:.2f}, Actual: {actual}")
                            print(f"         Explanation: {explanation}")
                    else:
                        print(f"No movies found matching '{search_term}'")
                
                elif detail_input == 'new':
                    break
                
                else:
                    print("Invalid option. Use: top50, bottom20, rated, search, or new")
        
        except ValueError:
            print("❌ Please enter a valid user ID (number)")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    interactive_demo()