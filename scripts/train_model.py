import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessor import DataPreprocessor
from src.model import RecommendationModel, BaselineModel
from src.ranker import MovieRanker
from src.collaborative_filter import GraphBasedCollaborativeFilter
from config.settings import *
import joblib

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv(COMBINED_DATA_PATH)
    
    # Initialize components
    preprocessor = DataPreprocessor()
    ml_model = RecommendationModel('xgb')
    baseline_model = BaselineModel()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features available: {list(df.columns)}")
    
    # Prepare features
    print("\nPreparing features...")
    X, y = preprocessor.prepare_features(df, is_training=True)
    
    # Show final feature list
    print(f"\nFinal Feature List ({X.shape[1]} features):")
    print("=" * 60)
    for i, feature in enumerate(X.columns, 1):
        print(f"{i:2d}. {feature}")
    
    print(f"\nFeature Categories:")
    basic_features = [col for col in X.columns if col in ['user_encoded', 'item_id', 'age', 'gender_encoded', 'occupation_encoded', 'user_age_at_release']]
    rating_features = [col for col in X.columns if 'rating' in col]
    genre_binary = [col for col in X.columns if col in ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']]
    
    print(f"Basic features ({len(basic_features)}): {basic_features}")
    print(f"Rating features ({len(rating_features)}): {rating_features[:5]}...")
    print(f"Binary genre features ({len(genre_binary)}): {genre_binary}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    print("Training Random Forest model...")
    ml_model = RecommendationModel('rf')
    ml_model.train(X_train, y_train)
    
    print("Training baseline model...")
    train_df = df.iloc[X_train.index]
    baseline_model.train(train_df)
    
    # Evaluate models
    print("\nModel Evaluation:")
    ml_metrics = ml_model.evaluate(X_test, y_test)
    print(f"ML Model - RMSE: {ml_metrics['rmse']:.3f}, MAE: {ml_metrics['mae']:.3f}")
    
    # Baseline evaluation
    test_df = df.iloc[X_test.index]
    baseline_preds = [baseline_model.predict(row['user_id'], row['item_id']) 
                     for _, row in test_df.iterrows()]
    baseline_rmse = np.sqrt(np.mean((y_test - baseline_preds) ** 2))
    print(f"Baseline Model - RMSE: {baseline_rmse:.3f}")
    
    # Save models and preprocessor
    print("\nSaving models...")
    ml_model.save('models/ml_model.pkl')
    preprocessor.save('models/preprocessor.pkl')
    joblib.dump(baseline_model, 'models/baseline_model.pkl')
    
    # Save movie metadata for API
    movie_metadata = df[['item_id', 'title', 'year', 'genres']].drop_duplicates()
    movie_metadata.to_csv('models/movie_metadata.csv', index=False)
    
    print("Models saved successfully!")
    
    # Create ranker with hybrid system
    ranker = MovieRanker(ml_model, preprocessor, baseline_model)
    ranker.set_movie_data(df)
    
    # Save collaborative filter
    joblib.dump(ranker.collaborative_filter, 'models/collaborative_filter.pkl')
    
    # Demo ranking with actual ratings comparison
    user_id = 685
    print(f"\nDemo Ranking for User {user_id}:")
    candidate_movies = [56, 592, 1582, 171, 580, 1409, 953, 994, 387]
    
    # Get actual ratings for comparison
    user_ratings = df[df['user_id'] == user_id]
    actual_ratings = {}
    for movie_id in candidate_movies:
        rating_data = user_ratings[user_ratings['item_id'] == movie_id]
        if not rating_data.empty:
            actual_ratings[movie_id] = rating_data['rating'].iloc[0]
    
    hybrid_ranking = ranker.rank_movies(user_id, candidate_movies, df, use_baseline=False)
    baseline_ranking = ranker.rank_movies(user_id, candidate_movies, df, use_baseline=True)
    
    print("\nRanking Comparison:")
    print("=" * 90)
    print(f"{'Movie':<8} {'Hybrid':<8} {'Method':<12} {'Baseline':<8} {'Actual':<8} {'Title':<20}")
    print("=" * 90)
    
    # Create lookup dictionaries
    hybrid_lookup = {r['movie_id']: r['predicted_rating'] for r in hybrid_ranking}
    method_lookup = {r['movie_id']: r['method'] for r in hybrid_ranking}
    baseline_lookup = {r['movie_id']: r['predicted_rating'] for r in baseline_ranking}
    title_lookup = {r['movie_id']: r['title'] for r in hybrid_ranking}
    
    for movie_id in candidate_movies:
        hybrid_pred = hybrid_lookup.get(movie_id, 0)
        method = method_lookup.get(movie_id, 'unknown')[:11]
        baseline_pred = baseline_lookup.get(movie_id, 0)
        actual = actual_ratings.get(movie_id, 'N/A')
        title = title_lookup.get(movie_id, 'Unknown')[:20]
        
        print(f"{movie_id:<8} {hybrid_pred:<8.2f} {method:<12} {baseline_pred:<8.2f} {actual:<8} {title:<20}")
    
    print("\nTop 5 Hybrid Model Rankings:")
    for i, movie in enumerate(hybrid_ranking[:5]):
        actual = actual_ratings.get(movie['movie_id'], 'N/A')
        method = movie.get('method', 'unknown')
        print(f"{i+1}. {movie['title']} - Predicted: {movie['predicted_rating']:.2f} ({method}), Actual: {actual}")
    
    print("\nTop 5 Baseline Rankings:")
    for i, movie in enumerate(baseline_ranking[:5]):
        actual = actual_ratings.get(movie['movie_id'], 'N/A')
        print(f"{i+1}. {movie['title']} - Predicted: {movie['predicted_rating']:.2f}, Actual: {actual}")
    
    return ranker, ml_model, baseline_model, preprocessor

if __name__ == "__main__":
    ranker, ml_model, baseline_model, preprocessor = main()