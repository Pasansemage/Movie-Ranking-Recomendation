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
    
    # Prepare features
    print("Preparing features...")
    X, y = preprocessor.prepare_features(df, is_training=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    print("Training ML model...")
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
    
    # Demo ranking
    print("\nDemo Ranking for User 196:")
    candidate_movies = [1, 50, 100, 181, 258, 286, 288, 294, 300]
    
    ml_ranking = ranker.rank_movies(196, candidate_movies, df, use_baseline=False)
    baseline_ranking = ranker.rank_movies(196, candidate_movies, df, use_baseline=True)
    
    print("\nML Model Rankings:")
    for explanation in ranker.explain_ranking(ml_ranking):
        print(explanation)
    
    print("\nBaseline Model Rankings:")
    for explanation in ranker.explain_ranking(baseline_ranking):
        print(explanation)
    
    return ranker, ml_model, baseline_model, preprocessor

if __name__ == "__main__":
    ranker, ml_model, baseline_model, preprocessor = main()