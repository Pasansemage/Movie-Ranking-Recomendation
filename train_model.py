import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.preprocessor import DataPreprocessor
from src.model import RecommendationModel, BaselineModel
from src.ranker import MovieRanker
import joblib

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('ml100k_combined.csv')
    df['genres'] = df['genres'].fillna('[]')
    
    # Initialize components
    preprocessor = DataPreprocessor()
    ml_model = RecommendationModel('rf')
    baseline_model = BaselineModel()
    
    # Calculate user weights for balanced training
    user_counts = df['user_id'].value_counts()
    df['user_weight'] = df['user_id'].map(lambda x: 1.0 / user_counts[x])
    
    print(f"User activity distribution:")
    print(f"Users with 20-50 ratings: {sum((user_counts >= 20) & (user_counts <= 50))}")
    print(f"Users with 51-100 ratings: {sum((user_counts >= 51) & (user_counts <= 100))}")
    print(f"Users with 100+ ratings: {sum(user_counts > 100)}")
    
    # Prepare features
    print("\nPreparing features...")
    X, y = preprocessor.prepare_features(df, is_training=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_weights = df.iloc[X_train.index]['user_weight'].values
    
    # Train models
    print("Training XGBoost model...")
    ml_model = RecommendationModel('xgb')
    ml_model.train(X_train, y_train)
    
    print("Training baseline model...")
    baseline_model.train(df)
    
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
    movie_metadata = df[['item_id', 'title', 'genres', 'year']].drop_duplicates()
    movie_metadata.to_csv('models/movie_metadata.csv', index=False)
    
    print("Models saved successfully!")
    
    # Create ranker
    ranker = MovieRanker(ml_model, preprocessor, baseline_model)
    ranker.set_movie_data(df)
    
    # Demo ranking with actual ratings comparison
    print("\nDemo Ranking for User 196:")
    candidate_movies = [1, 50, 100, 181, 258, 286, 288, 294, 300]
    
    # Get actual ratings for comparison
    user_196_ratings = df[df['user_id'] == 196]
    actual_ratings = {}
    for movie_id in candidate_movies:
        rating_data = user_196_ratings[user_196_ratings['item_id'] == movie_id]
        if not rating_data.empty:
            actual_ratings[movie_id] = rating_data['rating'].iloc[0]
    
    ml_ranking = ranker.rank_movies(196, candidate_movies, df, use_baseline=False)
    baseline_ranking = ranker.rank_movies(196, candidate_movies, df, use_baseline=True)
    
    print("\nRanking Comparison:")
    print("=" * 80)
    print(f"{'Movie':<35} {'ML Pred':<8} {'Baseline':<8} {'Actual':<8} {'Title':<20}")
    print("=" * 80)
    
    # Create lookup dictionaries
    ml_lookup = {r['movie_id']: r['predicted_rating'] for r in ml_ranking}
    baseline_lookup = {r['movie_id']: r['predicted_rating'] for r in baseline_ranking}
    title_lookup = {r['movie_id']: r['title'] for r in ml_ranking}
    
    for movie_id in candidate_movies:
        ml_pred = ml_lookup.get(movie_id, 0)
        baseline_pred = baseline_lookup.get(movie_id, 0)
        actual = actual_ratings.get(movie_id, 'N/A')
        title = title_lookup.get(movie_id, 'Unknown')[:20]
        
        print(f"{movie_id:<35} {ml_pred:<8.2f} {baseline_pred:<8.2f} {actual:<8} {title:<20}")
    
    print("\nTop 5 ML Model Rankings:")
    for i, movie in enumerate(ml_ranking[:5]):
        actual = actual_ratings.get(movie['movie_id'], 'N/A')
        print(f"{i+1}. {movie['title']} - Predicted: {movie['predicted_rating']:.2f}, Actual: {actual}")
    
    print("\nTop 5 Baseline Rankings:")
    for i, movie in enumerate(baseline_ranking[:5]):
        actual = actual_ratings.get(movie['movie_id'], 'N/A')
        print(f"{i+1}. {movie['title']} - Predicted: {movie['predicted_rating']:.2f}, Actual: {actual}")
    
    # Calculate ranking accuracy if we have actual ratings
    if actual_ratings:
        print("\nRanking Accuracy Analysis:")
        # Sort actual ratings by rating value (descending)
        actual_sorted = sorted(actual_ratings.items(), key=lambda x: x[1], reverse=True)
        actual_order = [movie_id for movie_id, _ in actual_sorted]
        
        # Get predicted orders
        ml_order = [r['movie_id'] for r in ml_ranking if r['movie_id'] in actual_ratings]
        baseline_order = [r['movie_id'] for r in baseline_ranking if r['movie_id'] in actual_ratings]
        
        print(f"Actual rating order: {actual_order}")
        print(f"ML model order: {ml_order}")
        print(f"Baseline order: {baseline_order}")
    
    return ranker, ml_model, baseline_model, preprocessor

if __name__ == "__main__":
    ranker, ml_model, baseline_model, preprocessor = main()