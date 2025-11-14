import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import ast

class DataPreprocessor:
    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.occupation_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        # Statistical features from training set only
        self.global_avg_rating = 0
        self.user_avg_ratings = {}
        self.item_avg_ratings = {}
        self.genre_global_avg_ratings = {}
        self.user_genre_avg_ratings = {}
        
    def parse_genres(self, genre_str):
        """Parse genre string to list"""
        try:
            return ast.literal_eval(genre_str)
        except:
            return []
    
    def prepare_features(self, df, is_training=True):
        """Prepare features with proper train/test separation"""
        df = df.copy()
        
        # Parse genres - handle missing genres column
        if 'genres' in df.columns:
            df['genre_list'] = df['genres'].apply(self.parse_genres)
        else:
            df['genre_list'] = [[] for _ in range(len(df))]
        
        # Calculate statistical features from training data only
        if is_training:
            self.global_avg_rating = df['rating'].mean()
            self.user_avg_ratings = df.groupby('user_id')['rating'].mean().to_dict()
            self.item_avg_ratings = df.groupby('item_id')['rating'].mean().to_dict()
            
            # Genre-wise global average ratings
            genre_names = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
                          'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 
                          'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
            
            self.genre_global_avg_ratings = {}
            for genre in genre_names:
                genre_ratings = []
                for _, row in df.iterrows():
                    if genre in row['genre_list']:
                        genre_ratings.append(row['rating'])
                self.genre_global_avg_ratings[genre] = np.mean(genre_ratings) if genre_ratings else self.global_avg_rating
            
            # User-wise genre average ratings
            self.user_genre_avg_ratings = {}
            for user_id in df['user_id'].unique():
                user_data = df[df['user_id'] == user_id]
                self.user_genre_avg_ratings[user_id] = {}
                for genre in genre_names:
                    genre_ratings = []
                    for _, row in user_data.iterrows():
                        if genre in row['genre_list']:
                            genre_ratings.append(row['rating'])
                    self.user_genre_avg_ratings[user_id][genre] = np.mean(genre_ratings) if genre_ratings else self.user_avg_ratings.get(user_id, self.global_avg_rating)
        
        # Basic features
        if is_training:
            df['user_encoded'] = self.user_encoder.fit_transform(df['user_id'])
            df['occupation_encoded'] = self.occupation_encoder.fit_transform(df['occupation'])
        else:
            df['user_encoded'] = self.user_encoder.transform(df['user_id'])
            df['occupation_encoded'] = self.occupation_encoder.transform(df['occupation'])
            
        df['gender_encoded'] = (df['gender'] == 'M').astype(int)
        
        # User age when movie was released (assuming data collected Jan 1998)
        df['user_age_at_release'] = df['age'] - (1998 - df['year'].fillna(1998))
        
        # Statistical features using training set statistics
        df['movie_global_avg_rating'] = df['item_id'].map(self.item_avg_ratings).fillna(self.global_avg_rating)
        df['user_avg_rating'] = df['user_id'].map(self.user_avg_ratings).fillna(self.global_avg_rating)
        
        # Binary genre columns
        genre_names = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
                      'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 
                      'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        
        for genre in genre_names:
            df[genre] = df['genre_list'].apply(lambda x: 1 if genre in x else 0)
        
        # Genre-wise global average ratings
        for genre in genre_names:
            df[f'global_avg_{genre.lower()}_rating'] = self.genre_global_avg_ratings.get(genre, self.global_avg_rating)
        
        # User-wise genre average ratings
        for genre in genre_names:
            df[f'user_avg_{genre.lower()}_rating'] = df.apply(
                lambda row: self.user_genre_avg_ratings.get(row['user_id'], {}).get(genre, self.global_avg_rating), axis=1
            )
        
        # Select feature columns
        feature_cols = ['user_encoded', 'item_id', 'age', 'gender_encoded', 'occupation_encoded', 
                       'user_age_at_release', 'movie_global_avg_rating', 'user_avg_rating']
        
        # Add binary genre columns
        feature_cols.extend(genre_names)
        
        # Add genre average rating columns
        feature_cols.extend([f'global_avg_{genre.lower()}_rating' for genre in genre_names])
        feature_cols.extend([f'user_avg_{genre.lower()}_rating' for genre in genre_names])
        
        features = df[feature_cols].copy()
        
        # Scale numerical features
        num_cols = ['age', 'user_age_at_release', 'movie_global_avg_rating', 'user_avg_rating'] + \
                   [f'global_avg_{genre.lower()}_rating' for genre in genre_names] + \
                   [f'user_avg_{genre.lower()}_rating' for genre in genre_names]
        
        if is_training:
            features[num_cols] = self.scaler.fit_transform(features[num_cols])
        else:
            features[num_cols] = self.scaler.transform(features[num_cols])
        
        return features, df['rating'] if 'rating' in df.columns else None
    
    def save(self, filepath):
        """Save preprocessor"""
        joblib.dump({
            'user_encoder': self.user_encoder,
            'occupation_encoder': self.occupation_encoder,
            'scaler': self.scaler,
            'global_avg_rating': self.global_avg_rating,
            'user_avg_ratings': self.user_avg_ratings,
            'item_avg_ratings': self.item_avg_ratings,
            'genre_global_avg_ratings': self.genre_global_avg_ratings,
            'user_genre_avg_ratings': self.user_genre_avg_ratings
        }, filepath)
    
    def load(self, filepath):
        """Load preprocessor"""
        components = joblib.load(filepath)
        self.user_encoder = components['user_encoder']
        self.occupation_encoder = components['occupation_encoder']
        self.scaler = components['scaler']
        self.global_avg_rating = components.get('global_avg_rating', 0)
        self.user_avg_ratings = components.get('user_avg_ratings', {})
        self.item_avg_ratings = components.get('item_avg_ratings', {})
        self.genre_global_avg_ratings = components.get('genre_global_avg_ratings', {})
        self.user_genre_avg_ratings = components.get('user_genre_avg_ratings', {})
        return self