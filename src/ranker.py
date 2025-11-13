import pandas as pd
import numpy as np

class MovieRanker:
    def __init__(self, model, preprocessor, baseline_model=None):
        self.model = model
        self.preprocessor = preprocessor
        self.baseline_model = baseline_model
        self.movie_data = None
        
    def set_movie_data(self, df):
        """Set movie metadata for explanations"""
        self.movie_data = df[['item_id', 'title', 'genres', 'year']].drop_duplicates()
        
    def rank_movies(self, user_id, candidate_movies, df, use_baseline=False):
        """Rank candidate movies for a given user"""
        # Filter data for the user and candidate movies
        user_data = df[df['user_id'] == user_id].iloc[0:1]  # Get user info
        
        rankings = []
        for movie_id in candidate_movies:
            if use_baseline and self.baseline_model:
                score = self.baseline_model.predict(user_id, movie_id)
            else:
                # Create feature vector for this user-movie pair
                movie_data = df[df['item_id'] == movie_id].iloc[0:1]
                test_row = user_data.copy()
                test_row['item_id'] = movie_id
                test_row['title'] = movie_data['title'].iloc[0]
                test_row['genres'] = movie_data['genres'].iloc[0]
                test_row['year'] = movie_data['year'].iloc[0]
                test_row['movie_age'] = 1998 - test_row['year'].fillna(1998)
                
                # Prepare features for inference
                features, _ = self.preprocessor.prepare_features(test_row, is_training=False)
                score = self.model.predict(features)[0]
            
            # Get movie info for explanation
            movie_info = self.movie_data[self.movie_data['item_id'] == movie_id].iloc[0]
            
            rankings.append({
                'movie_id': movie_id,
                'title': movie_info['title'],
                'predicted_rating': score,
                'genres': movie_info['genres'],
                'year': movie_info['year']
            })
        
        # Sort by predicted rating (descending)
        rankings.sort(key=lambda x: x['predicted_rating'], reverse=True)
        
        return rankings
    
    def explain_ranking(self, ranking):
        """Provide explanation for why movies are ranked highly"""
        explanations = []
        for i, movie in enumerate(ranking[:5]):  # Top 5 explanations
            explanation = f"#{i+1}: {movie['title']} (Rating: {movie['predicted_rating']:.2f})"
            explanation += f" - {movie['genres']} from {movie['year']}"
            explanations.append(explanation)
        return explanations