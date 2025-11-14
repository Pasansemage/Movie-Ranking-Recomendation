import pandas as pd
import numpy as np

class MovieRanker:
    def __init__(self, model, preprocessor, baseline_model=None):
        self.model = model
        self.preprocessor = preprocessor
        self.baseline_model = baseline_model
        self.movie_data = None
        
    def set_movie_data(self, df):
        """Set movie metadata"""
        self.movie_data = df[['item_id', 'title', 'year']].drop_duplicates()
        # Add genres as list if available
        if 'genres' in df.columns:
            genre_data = df[['item_id', 'genres']].drop_duplicates()
            self.movie_data = self.movie_data.merge(genre_data, on='item_id', how='left')
        print("Movie data loaded successfully!")
        
    def rank_movies(self, user_id, candidate_movies, df, use_baseline=False):
        """Rank candidate movies using ML model with new features"""
        rankings = []
        
        # Get user profile for feature creation
        user_profile = df[df['user_id'] == user_id].iloc[0]
        
        for movie_id in candidate_movies:
            if use_baseline and self.baseline_model:
                score = self.baseline_model.predict(user_id, movie_id)
                method = 'baseline'
            else:
                # Create synthetic user-movie pair with all features
                movie_data = df[df['item_id'] == movie_id].iloc[0]
                
                # Create test row with all required features
                test_row = pd.DataFrame([{
                    'user_id': user_id,
                    'item_id': movie_id,
                    'age': user_profile['age'],
                    'gender': user_profile['gender'],
                    'occupation': user_profile['occupation'],
                    'user_age_at_release': user_profile.get('user_age_at_release', user_profile['age']),
                    'movie_avg_rating': movie_data.get('movie_avg_rating', 3.5),
                    'user_avg_rating': user_profile.get('user_avg_rating', 3.5),
                    'title': movie_data['title'],
                    'year': movie_data['year']
                }])
                
                # Add binary genre columns
                genre_names = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
                              'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 
                              'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
                
                for genre in genre_names:
                    test_row[genre] = movie_data.get(genre, 0)
                    test_row[f'user_avg_{genre.lower()}_rating'] = user_profile.get(f'user_avg_{genre.lower()}_rating', user_profile.get('user_avg_rating', 3.5))
                    test_row[f'global_avg_{genre.lower()}_rating'] = movie_data.get(f'global_avg_{genre.lower()}_rating', 3.5)
                
                features, _ = self.preprocessor.prepare_features(test_row, is_training=False)
                score = self.model.predict(features)[0]
                method = 'ml_model'
            
            # Get movie info
            movie_info = self.movie_data[self.movie_data['item_id'] == movie_id].iloc[0]
            
            rankings.append({
                'movie_id': movie_id,
                'title': movie_info['title'],
                'predicted_rating': score,
                'genres': movie_info.get('genres', []),
                'year': movie_info['year'],
                'method': method
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