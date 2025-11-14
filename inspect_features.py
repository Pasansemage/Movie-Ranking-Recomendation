import pandas as pd
import numpy as np
from src.preprocessor import DataPreprocessor
from src.model import RecommendationModel

def inspect_features():
    """Inspect the features created by the preprocessor"""
    
    # Load data
    df = pd.read_csv('ml100k_combined.csv')
    df['genres'] = df['genres'].fillna('[]')
    
    # Take a small sample for inspection
    sample_df = df.head(100).copy()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Prepare features
    print("Creating features...")
    X, y = preprocessor.prepare_features(sample_df, is_training=True)
    
    print(f"\nFeature Matrix Shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")
    
    # Show feature columns
    print(f"\nFeature Columns:")
    for i, col in enumerate(X.columns):
        print(f"{i:3d}: {col}")
    
    # Show sample of original vs processed data
    print(f"\nSample Original Data:")
    print(sample_df[['user_id', 'item_id', 'age', 'gender', 'occupation', 'genres', 'year']].head(3))
    
    print(f"\nSample Processed Features:")
    print(X.head(3))
    
    # Show encoding mappings
    print(f"\nUser Encoding Sample:")
    user_mapping = dict(zip(preprocessor.user_encoder.classes_, 
                           preprocessor.user_encoder.transform(preprocessor.user_encoder.classes_)))
    print(f"First 10 users: {dict(list(user_mapping.items())[:10])}")
    
    print(f"\nOccupation Encoding:")
    occ_mapping = dict(zip(preprocessor.occupation_encoder.classes_, 
                          preprocessor.occupation_encoder.transform(preprocessor.occupation_encoder.classes_)))
    print(occ_mapping)
    
    # Show genre features
    individual_genre_features = [col for col in X.columns if col.startswith('has_')]
    tfidf_genre_features = [col for col in X.columns if col.startswith('tfidf_')]
    print(f"\nIndividual Genre Features: {len(individual_genre_features)} total")
    print(f"Individual genre features: {individual_genre_features}")
    print(f"\nTF-IDF Genre Features: {len(tfidf_genre_features)} total")
    print(f"Sample TF-IDF features: {tfidf_genre_features[:10]}")
    
    # Show feature statistics
    print(f"\nFeature Statistics:")
    print(X.describe())
    
    # Train model to get feature importance
    print(f"\nTraining model for feature importance...")
    model = RecommendationModel('rf')
    model.train(X, y)
    
    # Get feature importance
    if hasattr(model.model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 20 Most Important Features:")
        print(importance_df.head(20))
        
        print(f"\nFeature Importance by Category:")
        
        # Categorize features
        user_features = importance_df[importance_df['feature'].isin(['user_encoded', 'age', 'gender_encoded', 'occupation_encoded'])]
        movie_features = importance_df[importance_df['feature'].isin(['item_id', 'movie_age'])]
        individual_genre_features = importance_df[importance_df['feature'].str.startswith('has_')]
        tfidf_genre_features = importance_df[importance_df['feature'].str.startswith('tfidf_')]
        
        print(f"\nUser Features Importance:")
        print(user_features)
        
        print(f"\nMovie Features Importance:")
        print(movie_features)
        
        print(f"\nTop 10 Individual Genre Features Importance:")
        print(individual_genre_features.head(10))
        
        print(f"\nTop 10 TF-IDF Genre Features Importance:")
        print(tfidf_genre_features.head(10))
        
        # Summary by category
        print(f"\nImportance Summary:")
        print(f"User features total importance: {user_features['importance'].sum():.3f}")
        print(f"Movie features total importance: {movie_features['importance'].sum():.3f}")
        print(f"Individual genre features total importance: {individual_genre_features['importance'].sum():.3f}")
        print(f"TF-IDF genre features total importance: {tfidf_genre_features['importance'].sum():.3f}")
        print(f"All genre features total importance: {(individual_genre_features['importance'].sum() + tfidf_genre_features['importance'].sum()):.3f}")

if __name__ == "__main__":
    inspect_features()