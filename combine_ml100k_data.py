import pandas as pd
import numpy as np
from datetime import datetime

def combine_ml100k_data():
    # Load ratings data
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    
    # Load user data
    users = pd.read_csv('ml-100k/u.user', sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
    
    # Load item data
    genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
                  'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
                  'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    
    items = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', 
                       names=['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + genre_cols)
    
    # Clean item data
    items['release_date'] = pd.to_datetime(items['release_date'], format='%d-%b-%Y', errors='coerce')
    items['year'] = items['release_date'].dt.year
    
    # Create combined dataset
    combined = ratings.merge(users, on='user_id', how='left')
    combined = combined.merge(items, on='item_id', how='left')
    
    # Add genre information as a list
    combined['genres'] = combined[genre_cols].apply(
        lambda x: [genre for genre, val in zip(genre_cols, x) if val == 1], axis=1
    )
    
    # Drop individual genre columns from combined dataset
    combined = combined.drop(columns=genre_cols + ['video_release_date', 'imdb_url'])
    
    # Save combined dataset
    combined.to_csv('ml100k_combined.csv', index=False)
    
    # Create summary statistics
    summary = {
        'total_ratings': len(combined),
        'unique_users': combined['user_id'].nunique(),
        'unique_items': combined['item_id'].nunique(),
        'rating_range': f"{combined['rating'].min()}-{combined['rating'].max()}",
        'date_range': f"{combined['timestamp'].min()} to {combined['timestamp'].max()}",
        'avg_rating': combined['rating'].mean(),
        'most_common_genres': items[genre_cols].sum().sort_values(ascending=False).head(5).to_dict()
    }
    
    print("MovieLens 100k Data Combined Successfully!")
    print(f"Dataset shape: {combined.shape}")
    print(f"Columns: {list(combined.columns)}")
    print("\nSummary Statistics:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    return combined

if __name__ == "__main__":
    combined_data = combine_ml100k_data()