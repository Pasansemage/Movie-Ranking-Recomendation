import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

class GraphBasedCollaborativeFilter:
    def __init__(self, min_similarity=0.3, min_neighbors=5):
        self.min_similarity = min_similarity
        self.min_neighbors = min_neighbors
        self.user_similarity_matrix = None
        self.user_profiles = None
        self.rating_matrix = None
        
    def build_user_graph(self, df):
        """Build user similarity graph based on demographics and movie preferences"""
        users = df[['user_id', 'age', 'gender', 'occupation', 'zip_code']].drop_duplicates()
        
        # Create user-movie rating matrix
        rating_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating', fill_value=0)
        self.rating_matrix = rating_matrix
        
        # Create demographic features
        user_features = []
        for _, user in users.iterrows():
            # Age normalization
            age_norm = user['age'] / 100.0
            
            # Gender encoding
            gender_enc = 1 if user['gender'] == 'M' else 0
            
            # Location similarity (first 2 digits of zip code)
            zip_str = str(user['zip_code'])[:2]
            try:
                location = int(zip_str) if len(zip_str) >= 2 and zip_str.isdigit() else 0
            except ValueError:
                location = 0
            location_norm = location / 100.0
            
            # Movie preference profile (genre preferences)
            user_ratings = df[df['user_id'] == user['user_id']]
            genre_prefs = self._calculate_genre_preferences(user_ratings)
            
            # Combine features
            features = [age_norm, gender_enc, location_norm] + genre_prefs
            user_features.append(features)
        
        # Calculate similarity matrix
        scaler = StandardScaler()
        user_features_scaled = scaler.fit_transform(user_features)
        similarity_matrix = cosine_similarity(user_features_scaled)
        
        # Create user mapping
        user_ids = users['user_id'].values
        self.user_similarity_matrix = pd.DataFrame(
            similarity_matrix, 
            index=user_ids, 
            columns=user_ids
        )
        
        return self.user_similarity_matrix
    
    def _calculate_genre_preferences(self, user_ratings):
        """Calculate user's genre preferences based on ratings"""
        genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
                 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 
                 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        
        genre_scores = []
        for genre in genres:
            # Get movies with this genre that user rated
            genre_movies = user_ratings[user_ratings['genres'].str.contains(genre, na=False)]
            if len(genre_movies) > 0:
                avg_rating = genre_movies['rating'].mean()
                genre_scores.append(avg_rating / 5.0)  # Normalize to 0-1
            else:
                genre_scores.append(0.0)
        
        return genre_scores
    
    def find_similar_users(self, user_id, k=10):
        """Find k most similar users to given user"""
        if self.user_similarity_matrix is None:
            return []
        
        if user_id not in self.user_similarity_matrix.index:
            return []
        
        similarities = self.user_similarity_matrix.loc[user_id]
        # Remove self and filter by minimum similarity
        similarities = similarities[similarities.index != user_id]
        similarities = similarities[similarities >= self.min_similarity]
        
        # Return top k similar users
        return similarities.nlargest(k).to_dict()
    
    def predict_rating(self, user_id, item_id, df):
        """Predict rating using collaborative filtering"""
        similar_users = self.find_similar_users(user_id)
        
        if len(similar_users) < self.min_neighbors:
            return None  # Not enough similar users, use ML model
        
        # Get ratings from similar users for this item
        weighted_ratings = []
        total_weight = 0
        
        for similar_user_id, similarity in similar_users.items():
            user_item_rating = df[(df['user_id'] == similar_user_id) & 
                                (df['item_id'] == item_id)]
            
            if not user_item_rating.empty:
                rating = user_item_rating['rating'].iloc[0]
                weighted_ratings.append(rating * similarity)
                total_weight += similarity
        
        if total_weight == 0:
            return None  # No similar users rated this item
        
        predicted_rating = sum(weighted_ratings) / total_weight
        return np.clip(predicted_rating, 1, 5)  # Ensure rating is in valid range

class HybridRecommender:
    def __init__(self, collaborative_filter, ml_model, preprocessor):
        self.collaborative_filter = collaborative_filter
        self.ml_model = ml_model
        self.preprocessor = preprocessor
        
    def predict_rating(self, user_id, item_id, df):
        """Hybrid prediction: CF first, then ML model fallback"""
        # Try collaborative filtering first
        cf_prediction = self.collaborative_filter.predict_rating(user_id, item_id, df)
        
        if cf_prediction is not None:
            return cf_prediction, 'collaborative'
        
        # Fallback to ML model
        user_data = df[(df['user_id'] == user_id) & (df['item_id'] == item_id)]
        if user_data.empty:
            # Create synthetic user-item pair for prediction
            user_profile = df[df['user_id'] == user_id].iloc[0:1].copy()
            item_profile = df[df['item_id'] == item_id].iloc[0:1].copy()
            
            synthetic_data = user_profile.copy()
            synthetic_data['item_id'] = item_id
            synthetic_data['title'] = item_profile['title'].iloc[0]
            synthetic_data['genres'] = item_profile['genres'].iloc[0]
            synthetic_data['year'] = item_profile['year'].iloc[0]
        else:
            synthetic_data = user_data
        
        X, _ = self.preprocessor.prepare_features(synthetic_data, is_training=False)
        ml_prediction = self.ml_model.predict(X)[0]
        
        return np.clip(ml_prediction, 1, 5), 'ml_model'