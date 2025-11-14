import pandas as pd
import joblib
from src.preprocessor import DataPreprocessor
from src.model import RecommendationModel, BaselineModel
from src.ranker import MovieRanker

def demo_recommendation_system():
    """Demo script showing XGBoost vs Baseline model comparison with prediction methods"""
    
    print("Movie Recommendation System Demo")
    print("Using XGBoost model with advanced feature engineering")
    print("=" * 60)
    
    # Load pre-trained models
    print("Loading pre-trained models...")
    df = pd.read_csv('ml100k_combined.csv')
    
    try:
        # Load trained models
        preprocessor = DataPreprocessor().load('models/preprocessor.pkl')
        ml_model = RecommendationModel('xgb').load('models/ml_model.pkl')
        baseline_model = joblib.load('models/baseline_model.pkl')
        
        # Create ranker
        ranker = MovieRanker(ml_model, preprocessor, baseline_model)
        ranker.set_movie_data(df)
        
        print("Models loaded successfully!")
        
    except FileNotFoundError:
        print("Pre-trained models not found. Training new models...")
        # Fallback to training
        preprocessor = DataPreprocessor()
        X, y = preprocessor.prepare_features(df, is_training=True)
        
        ml_model = RecommendationModel('xgb')
        ml_model.train(X, y)
        
        baseline_model = BaselineModel()
        baseline_model.train(df)
        
        ranker = MovieRanker(ml_model, preprocessor, baseline_model)
        ranker.set_movie_data(df)
    
    # Demo with User 405
    user_id = 655
    candidate_movies = [56, 593, 158, 111, 582, 1400, 953, 995, 386]
    
    print(f"\nDemo: Ranking movies for User {user_id}")
    print("Candidate movies from user's actual ratings")
    print("=" * 60)
    
    # Get actual ratings for comparison
    user_ratings = df[df['user_id'] == user_id]
    actual_ratings = {}
    for movie_id in candidate_movies:
        rating_data = user_ratings[user_ratings['item_id'] == movie_id]
        if not rating_data.empty:
            actual_ratings[movie_id] = rating_data['rating'].iloc[0]
    
    # Get hybrid rankings (XGBoost + Collaborative Filtering)
    hybrid_rankings = ranker.rank_movies(user_id, candidate_movies, df, use_baseline=False)
    
    # Get baseline rankings
    baseline_rankings = ranker.rank_movies(user_id, candidate_movies, df, use_baseline=True)
    
    # Model Comparison
    print("\nModel Comparison:")
    print("=" * 80)
    print(f"{'Movie ID':<10} {'XGBoost':<8} {'Method':<12} {'Baseline':<8} {'Actual':<8} {'Title':<25}")
    print("=" * 80)
    
    # Create lookup dictionaries
    hybrid_lookup = {r['movie_id']: r['predicted_rating'] for r in hybrid_rankings}
    method_lookup = {r['movie_id']: r['method'] for r in hybrid_rankings}
    baseline_lookup = {r['movie_id']: r['predicted_rating'] for r in baseline_rankings}
    title_lookup = {r['movie_id']: r['title'] for r in hybrid_rankings}
    
    for movie_id in candidate_movies:
        hybrid_pred = hybrid_lookup.get(movie_id, 0)
        method = method_lookup.get(movie_id, 'unknown')[:11]
        baseline_pred = baseline_lookup.get(movie_id, 0)
        actual = actual_ratings.get(movie_id, 'N/A')
        title = title_lookup.get(movie_id, 'Unknown')[:24]
        
        print(f"{movie_id:<10} {hybrid_pred:<8.2f} {method:<12} {baseline_pred:<8.2f} {actual:<8} {title:<25}")
    
    # Top Recommendations
    print("\nTop 5 XGBoost Model Recommendations:")
    print("=" * 50)
    for i, movie in enumerate(hybrid_rankings[:5]):
        actual = actual_ratings.get(movie['movie_id'], 'N/A')
        method = movie.get('method', 'unknown')
        print(f"{i+1}. {movie['title']}")
        print(f"   Predicted: {movie['predicted_rating']:.2f} | Actual: {actual} | Method: {method}")
        print(f"   Year: {movie['year']}")
        print()
    
    print("Top 5 Baseline Model Recommendations:")
    print("=" * 50)
    for i, movie in enumerate(baseline_rankings[:5]):
        actual = actual_ratings.get(movie['movie_id'], 'N/A')
        print(f"{i+1}. {movie['title']}")
        print(f"   Predicted: {movie['predicted_rating']:.2f} | Actual: {actual}")
        print(f"   Year: {movie['year']}")
        print()
    
    # Feature Analysis
    print("Feature Engineering Summary:")
    print("=" * 40)
    print(f"• Dataset: {df.shape[0]:,} ratings, {df.shape[1]} features")
    print(f"• Binary genre features: 18 genres")
    print(f"• User age at movie release")
    print(f"• Movie average ratings")
    print(f"• User average ratings (overall + per genre)")
    print(f"• Global genre average ratings")
    print(f"• TF-IDF genre features")
    print(f"• Collaborative filtering for similar users")
    
    return ranker

if __name__ == "__main__":
    ranker = demo_recommendation_system()