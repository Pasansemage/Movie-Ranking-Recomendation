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
    
    # Create composite primary key
    combined['user_item_id'] = combined['user_id'].astype(str) + '_' + combined['item_id'].astype(str)
    
    # Keep binary genre columns (drop unknown)
    genre_cols_clean = [col for col in genre_cols if col != 'unknown']
    
    # Calculate reference date (Jan 1, 1998)
    reference_date = pd.to_datetime('1998-01-01')
    
    # Feature engineering
    # 1. User age at movie release
    combined['user_age_at_release'] = combined['age'] - ((reference_date - combined['release_date']).dt.days / 365.25)
    combined['user_age_at_release'] = combined['user_age_at_release'].fillna(combined['age'])
    
    # 2. Average rating for each movie
    movie_avg_ratings = combined.groupby('item_id')['rating'].mean().reset_index()
    movie_avg_ratings.columns = ['item_id', 'movie_avg_rating']
    combined = combined.merge(movie_avg_ratings, on='item_id', how='left')
    
    # 3. Average rating given out by each user
    user_avg_ratings = combined.groupby('user_id')['rating'].mean().reset_index()
    user_avg_ratings.columns = ['user_id', 'user_avg_rating']
    combined = combined.merge(user_avg_ratings, on='user_id', how='left')
    
    # 4. Average rating for each genre by each user
    for genre in genre_cols_clean:
        genre_mask = combined[genre] == 1
        user_genre_ratings = combined[genre_mask].groupby('user_id')['rating'].mean().reset_index()
        user_genre_ratings.columns = ['user_id', f'user_avg_{genre.lower()}_rating']
        combined = combined.merge(user_genre_ratings, on='user_id', how='left')
        combined[f'user_avg_{genre.lower()}_rating'] = combined[f'user_avg_{genre.lower()}_rating'].fillna(combined['user_avg_rating'])
    
    # 5. Average rating overall for each genre
    for genre in genre_cols_clean:
        genre_mask = combined[genre] == 1
        genre_avg = combined[genre_mask]['rating'].mean()
        combined[f'global_avg_{genre.lower()}_rating'] = genre_avg
    
    # Drop unnecessary columns
    combined = combined.drop(columns=['video_release_date', 'imdb_url', 'unknown'])
    
    # Save combined dataset
    combined.to_csv('ml100k_combined.csv', index=False)
    
    print("MovieLens 100k Data Combined Successfully!")
    print(f"Dataset shape: {combined.shape}")
    print(f"Columns: {list(combined.columns)}")
    print(f"Primary key: user_item_id (composite of user_id and item_id)")
    
    return combined

if __name__ == "__main__":
    combined_data = combine_ml100k_data()