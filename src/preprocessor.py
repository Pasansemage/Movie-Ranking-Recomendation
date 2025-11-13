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
        
        # Movie features
        df['movie_age'] = 1998 - df['year'].fillna(1998)
        df['genre_str'] = df['genres'].astype(str)
        
        # Genre features
        if is_training:
            genre_features = self.genre_vectorizer.fit_transform(df['genre_str'])
        else:
            genre_features = self.genre_vectorizer.transform(df['genre_str'])
            
        genre_df = pd.DataFrame(genre_features.toarray(), 
                               columns=[f'genre_{i}' for i in range(genre_features.shape[1])])
        
        # Combine features
        feature_cols = ['user_encoded', 'item_id', 'age', 'gender_encoded', 
                       'occupation_encoded', 'movie_age']
        features = df[feature_cols].copy()
        features = pd.concat([features.reset_index(drop=True), genre_df], axis=1)
        
        # Scale numerical features
        num_cols = ['age', 'movie_age']
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