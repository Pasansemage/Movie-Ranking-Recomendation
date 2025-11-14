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
            
            # Use user's average genre ratings from the new features
            user_data = df[df['user_id'] == user['user_id']].iloc[0]
            genre_names = ['action', 'adventure', 'animation', 'children', 'comedy', 'crime', 
                          'documentary', 'drama', 'fantasy', 'film-noir', 'horror', 'musical', 
                          'mystery', 'romance', 'sci-fi', 'thriller', 'war', 'western']
            
            genre_prefs = []
            for genre in genre_names:
                col_name = f'user_avg_{genre}_rating'
                if col_name in user_data:
                    genre_prefs.append(user_data[col_name] / 5.0)  # Normalize to 0-1
                else:
                    genre_prefs.append(0.0)
            
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
    
    def check_user_similarity(self, user_id, df, min_common_movies=10, max_rating_diff=0.5):
        """Check if user has similar preferences - returns count of similar users found"""
        similar_users = self.find_similar_users(user_id)
        
        if len(similar_users) >= self.min_neighbors:
            # Count users with sufficient common movies and similar ratings
            similar_count = 0
            total_common = 0
            
            user_ratings = df[df['user_id'] == user_id]
            
            for other_user_id in similar_users.keys():
                other_ratings = df[df['user_id'] == other_user_id]
                common_movies = set(user_ratings['item_id']) & set(other_ratings['item_id'])
                
                if len(common_movies) >= min_common_movies:
                    diffs = []
                    for movie_id in common_movies:
                        user_rating = user_ratings[user_ratings['item_id'] == movie_id]['rating'].iloc[0]
                        other_rating = other_ratings[other_ratings['item_id'] == movie_id]['rating'].iloc[0]
                        diffs.append(abs(user_rating - other_rating))
                    
                    avg_diff = np.mean(diffs)
                    if avg_diff <= max_rating_diff:
                        similar_count += 1
                        total_common += len(common_movies)
            
            if similar_count > 0:
                return True, similar_count, total_common, 0
        
        return False, 0, 0, 0
    
    def predict_rating(self, user_id, item_id, df):
        """Predict rating using collaborative filtering - requires at least 3 similar users"""
        similar_users = self.find_similar_users(user_id)
        
        if len(similar_users) < self.min_neighbors:
            return None  # Not enough similar users, use ML model
        
        # Get ratings from similar users for this item
        similar_ratings = []
        
        for similar_user_id, similarity in similar_users.items():
            user_item_rating = df[(df['user_id'] == similar_user_id) & 
                                (df['item_id'] == item_id)]
            
            if not user_item_rating.empty:
                rating = user_item_rating['rating'].iloc[0]
                similar_ratings.append(rating)
        
        # Require at least 3 similar users who rated this item
        if len(similar_ratings) < 3:
            return None  # Not enough similar users rated this item
        
        # Use simple average of similar users' ratings
        predicted_rating = np.mean(similar_ratings)
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
            synthetic_data['year'] = item_profile['year'].iloc[0]
            
            # Copy all genre columns and other features from item profile
            genre_names = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
                          'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 
                          'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
            
            for genre in genre_names:
                if genre in item_profile.columns:
                    synthetic_data[genre] = item_profile[genre].iloc[0]
                    
            # Copy other movie features
            movie_features = ['movie_avg_rating'] + [f'global_avg_{genre.lower()}_rating' for genre in genre_names]
            for feature in movie_features:
                if feature in item_profile.columns:
                    synthetic_data[feature] = item_profile[feature].iloc[0]
        else:
            synthetic_data = user_data
        
        X, _ = self.preprocessor.prepare_features(synthetic_data, is_training=False)
        ml_prediction = self.ml_model.predict(X)[0]
        
        return np.clip(ml_prediction, 1, 5), 'ml_model'