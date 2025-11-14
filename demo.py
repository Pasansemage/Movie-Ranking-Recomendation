import pandas as pd
from src.preprocessor import DataPreprocessor
from src.model import RecommendationModel, BaselineModel
from src.ranker import MovieRanker

def demo_recommendation_system():
    """Demo script showing how to use the recommendation system"""
    
    # Load data
    df = pd.read_csv('ml100k_combined.csv')
    df['genres'] = df['genres'].fillna('[]')
    
    # Initialize and train models
    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_features(df, is_training=True)
    
    ml_model = RecommendationModel('rf')
    ml_model.train(X, y)
    
    baseline_model = BaselineModel()
    baseline_model.train(df)
    
    # Create ranker
    ranker = MovieRanker(ml_model, preprocessor, baseline_model)
    ranker.set_movie_data(df)
    
    # Example usage
    user_id = 196
    candidate_movies = [1, 50, 100, 181, 258, 286, 288, 294, 300, 313]
    
    print(f"Ranking movies for User {user_id}")
    print("=" * 50)
    
    # Get rankings
    rankings = ranker.rank_movies(user_id, candidate_movies, df)
    
    # Display results
    print("Top Recommendations:")
    for i, movie in enumerate(rankings[:5]):
        print(f"{i+1}. {movie['title']} - Predicted Rating: {movie['predicted_rating']:.2f}")
        print(f"   Genres: {movie['genres']}, Year: {movie['year']}")
        print()
    
    return ranker

if __name__ == "__main__":
    ranker = demo_recommendation_system()