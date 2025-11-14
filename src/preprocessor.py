import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

class DataPreprocessor:
    def __init__(self):
        self.user_encoder = LabelEncoder()
        self.occupation_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.genre_vectorizer = TfidfVectorizer()
        
    def prepare_features(self, df, is_training=True):
        """Prepare features for training or inference"""
        df = df.copy()
        
        # User features
        if is_training:
            df['user_encoded'] = self.user_encoder.fit_transform(df['user_id'])
            df['occupation_encoded'] = self.occupation_encoder.fit_transform(df['occupation'])
        else:
            df['user_encoded'] = self.user_encoder.transform(df['user_id'])
            df['occupation_encoded'] = self.occupation_encoder.transform(df['occupation'])
            
        df['gender_encoded'] = (df['gender'] == 'M').astype(int)
        
        # Binary genre columns (already in data)
        genre_names = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
                      'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 
                      'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        
        # Create genre string for TF-IDF
        df['genre_str'] = df[genre_names].apply(
            lambda x: ' '.join([genre for genre, val in zip(genre_names, x) if val == 1]), axis=1
        )
        
        # TF-IDF genre features
        if is_training:
            genre_features = self.genre_vectorizer.fit_transform(df['genre_str'])
        else:
            genre_features = self.genre_vectorizer.transform(df['genre_str'])
            
        genre_tfidf_df = pd.DataFrame(genre_features.toarray(), 
                                     columns=[f'tfidf_{i}' for i in range(genre_features.shape[1])])
        
        # Feature columns
        feature_cols = ['user_encoded', 'item_id', 'user_age_at_release', 'gender_encoded', 
                       'occupation_encoded', 'movie_avg_rating', 'user_avg_rating']
        
        # Add binary genre columns
        feature_cols.extend(genre_names)
        
        # Add user genre average ratings
        user_genre_cols = [f'user_avg_{genre.lower()}_rating' for genre in genre_names]
        feature_cols.extend(user_genre_cols)
        
        # Add global genre average ratings
        global_genre_cols = [f'global_avg_{genre.lower()}_rating' for genre in genre_names]
        feature_cols.extend(global_genre_cols)
        
        # Select available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        features = df[available_cols].copy()
        
        # Add TF-IDF features
        features = pd.concat([features.reset_index(drop=True), genre_tfidf_df], axis=1)
        
        # Scale numerical features
        num_cols = ['user_age_at_release', 'movie_avg_rating', 'user_avg_rating'] + user_genre_cols + global_genre_cols
        num_cols = [col for col in num_cols if col in features.columns]
        
        if num_cols:
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
            'genre_vectorizer': self.genre_vectorizer
        }, filepath)
    
    def load(self, filepath):
        """Load preprocessor"""
        components = joblib.load(filepath)
        self.user_encoder = components['user_encoder']
        self.occupation_encoder = components['occupation_encoder']
        self.scaler = components['scaler']
        self.genre_vectorizer = components['genre_vectorizer']
        return self