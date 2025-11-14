import pandas as pd
import numpy as np
from src.preprocessor import DataPreprocessor

def inspect_features():
    # Load data
    print("Loading data...")
    df = pd.read_csv('ml100k_combined.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Sample a few rows for feature inspection
    sample_df = df.head(10)
    
    print("\nSample data:")
    print(sample_df[['user_id', 'item_id', 'rating', 'age', 'gender', 'occupation']].head())
    
    # Prepare features
    print("\nPreparing features...")
    X, y = preprocessor.prepare_features(sample_df, is_training=True)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Feature columns: {list(X.columns)}")
    
    # Show feature statistics
    print("\nFeature Statistics:")
    print("=" * 50)
    
    # Basic features
    basic_features = ['user_encoded', 'item_id', 'user_age_at_release', 'gender_encoded', 
                     'occupation_encoded', 'movie_avg_rating', 'user_avg_rating']
    
    print("Basic Features:")
    for feature in basic_features:
        if feature in X.columns:
            print(f"  {feature}: min={X[feature].min():.3f}, max={X[feature].max():.3f}, mean={X[feature].mean():.3f}")
    
    # Binary genre features
    genre_names = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
                  'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 
                  'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    
    print("\nBinary Genre Features:")
    for genre in genre_names:
        if genre in X.columns:
            count = X[genre].sum()
            print(f"  {genre}: {count} movies have this genre")
    
    # User genre average ratings
    print("\nUser Genre Average Ratings:")
    for genre in genre_names:
        col_name = f'user_avg_{genre.lower()}_rating'
        if col_name in X.columns:
            print(f"  {col_name}: min={X[col_name].min():.3f}, max={X[col_name].max():.3f}, mean={X[col_name].mean():.3f}")
    
    # Global genre average ratings
    print("\nGlobal Genre Average Ratings:")
    for genre in genre_names:
        col_name = f'global_avg_{genre.lower()}_rating'
        if col_name in X.columns:
            print(f"  {col_name}: {X[col_name].iloc[0]:.3f}")
    
    # TF-IDF features
    tfidf_cols = [col for col in X.columns if col.startswith('tfidf_')]
    print(f"\nTF-IDF Features: {len(tfidf_cols)} features")
    if tfidf_cols:
        tfidf_data = X[tfidf_cols]
        print(f"  Non-zero values: {(tfidf_data > 0).sum().sum()}")
        print(f"  Max value: {tfidf_data.max().max():.3f}")
        print(f"  Mean value: {tfidf_data.mean().mean():.3f}")
    
    # Show sample feature vector
    print("\nSample Feature Vector (first row):")
    print("=" * 50)
    sample_features = X.iloc[0]
    for i, (feature, value) in enumerate(sample_features.items()):
        if i < 20:  # Show first 20 features
            print(f"  {feature}: {value:.3f}")
        elif i == 20:
            print(f"  ... and {len(sample_features) - 20} more features")
            break
    
    # Feature importance analysis
    print("\nFeature Analysis:")
    print("=" * 50)
    
    # Check for missing values
    missing_counts = X.isnull().sum()
    if missing_counts.sum() > 0:
        print("Features with missing values:")
        for feature, count in missing_counts[missing_counts > 0].items():
            print(f"  {feature}: {count} missing values")
    else:
        print("No missing values in features")
    
    # Check feature variance
    print("\nFeature Variance Analysis:")
    variances = X.var()
    zero_var_features = variances[variances == 0].index.tolist()
    if zero_var_features:
        print(f"Zero variance features: {zero_var_features}")
    else:
        print("All features have non-zero variance")
    
    # Show correlation with target
    print("\nCorrelation with Rating:")
    correlations = X.corrwith(pd.Series(y, index=X.index)).abs().sort_values(ascending=False)
    print("Top 10 features correlated with rating:")
    for feature, corr in correlations.head(10).items():
        if not pd.isna(corr):
            print(f"  {feature}: {corr:.3f}")
    
    return X, y, preprocessor

if __name__ == "__main__":
    X, y, preprocessor = inspect_features()