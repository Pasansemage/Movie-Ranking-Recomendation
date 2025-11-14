import pandas as pd
import numpy as np
from .collaborative_filter import GraphBasedCollaborativeFilter, HybridRecommender

class MovieRanker:
    def __init__(self, model, preprocessor, baseline_model=None):
        self.model = model
        self.preprocessor = preprocessor
        self.baseline_model = baseline_model
        self.movie_data = None
        self.collaborative_filter = GraphBasedCollaborativeFilter()
        self.hybrid_recommender = None
        
    def set_movie_data(self, df):
        """Set movie metadata and build user similarity graph"""
        self.movie_data = df[['item_id', 'title', 'year', 'genres']].drop_duplicates()
        print("Building user similarity graph...")
        self.collaborative_filter.build_user_graph(df)
        self.hybrid_recommender = HybridRecommender(self.collaborative_filter, self.model, self.preprocessor)
        print("User graph built successfully!")
        
    def rank_movies(self, user_id, candidate_movies, df, use_baseline=False):
        """Rank candidate movies using hybrid approach with collaborative filtering for similar users"""
        rankings = []
        cf_count = 0
        ml_count = 0
        
        # Check if user has very similar preferences to other users
        has_similar, similar_count, total_common, _ = self.collaborative_filter.check_user_similarity(user_id, df)
        
        # Get user profile for feature creation
        user_profile = df[df['user_id'] == user_id].iloc[0]
        
        for movie_id in candidate_movies:
            current_features = None
            
            if use_baseline and self.baseline_model:
                score = self.baseline_model.predict(user_id, movie_id)
                method = 'baseline'
            elif has_similar and self.hybrid_recommender:
                # Use collaborative filtering for users with similar preferences
                score, method = self.hybrid_recommender.predict_rating(user_id, movie_id, df)
                if method == 'collaborative':
                    cf_count += 1
                else:
                    ml_count += 1
                    # Get features for ML explanation in hybrid mode
                    movie_data = df[df['item_id'] == movie_id].iloc[0]
                    test_row = pd.DataFrame([{
                        'user_id': user_id,
                        'item_id': movie_id,
                        'age': user_profile['age'],
                        'gender': user_profile['gender'],
                        'occupation': user_profile['occupation'],
                        'title': movie_data['title'],
                        'year': movie_data['year'],
                        'genres': movie_data['genres']
                    }])
                    current_features, _ = self.preprocessor.prepare_features(test_row, is_training=False)
            else:
                # Use ML model for users without similar preferences
                movie_data = df[df['item_id'] == movie_id].iloc[0]
                
                # Create test row with all required features
                test_row = pd.DataFrame([{
                    'user_id': user_id,
                    'item_id': movie_id,
                    'age': user_profile['age'],
                    'gender': user_profile['gender'],
                    'occupation': user_profile['occupation'],
                    'title': movie_data['title'],
                    'year': movie_data['year'],
                    'genres': movie_data['genres']
                }])
                
                current_features, _ = self.preprocessor.prepare_features(test_row, is_training=False)
                score = self.model.predict(current_features)[0]
                method = 'ml_model'
                ml_count += 1
            
            # Get movie info
            movie_info = self.movie_data[self.movie_data['item_id'] == movie_id].iloc[0]
            
            # Generate explanation
            if method == 'collaborative':
                similar_users = self.collaborative_filter.find_similar_users(user_id, k=5)
                user_count = len([u for u in similar_users.keys() if len(df[(df['user_id'] == u) & (df['item_id'] == movie_id)]) > 0])
                explanation = f"Recommended by {user_count} similar users"
            elif method == 'ml_model' and current_features is not None:
                # Get feature importance using model's feature_importances_
                try:
                    feature_names = current_features.columns.tolist()
                    importances = self.model.model.feature_importances_
                    
                    # Get top 3 most important features
                    top_indices = np.argsort(importances)[-3:][::-1]
                    top_features = []
                    
                    for idx in top_indices:
                        feature_name = feature_names[idx]
                        feature_value = current_features.iloc[0, idx]
                        
                        # Clean up feature names for explanation
                        if 'user_avg' in feature_name and 'rating' in feature_name:
                            clean_name = f"user's {feature_name.replace('user_avg_', '').replace('_rating', '')} preference"
                        elif 'movie_avg_rating' in feature_name:
                            clean_name = "movie popularity"
                        elif 'global_avg' in feature_name:
                            clean_name = f"general {feature_name.replace('global_avg_', '').replace('_rating', '')} appeal"
                        elif feature_name in ['age', 'user_age_at_release']:
                            clean_name = "age factor"
                        elif feature_name == 'year':
                            clean_name = "release year"
                        else:
                            clean_name = feature_name.replace('_', ' ')
                        
                        top_features.append(clean_name)
                    
                    explanation = f"Based on {', '.join(top_features[:2])}"
                except:
                    explanation = "ML model prediction based on user profile"
            else:
                explanation = "Baseline prediction"
            
            rankings.append({
                'movie_id': movie_id,
                'title': movie_info['title'],
                'predicted_rating': score,
                'year': movie_info['year'],
                'genres': movie_info['genres'],
                'method': method,
                'explanation': explanation
            })
        
        # Sort by predicted rating (descending)
        rankings.sort(key=lambda x: x['predicted_rating'], reverse=True)
        
        if has_similar:
            print(f"User {user_id} has similar preferences ({similar_count} similar users found with {total_common} total common movies)")
        
        if cf_count > 0 or ml_count > 0:
            print(f"Predictions: {cf_count} collaborative filtering (≥3 similar users rated movie), {ml_count} ML model")
        
        return rankings
    
    def explain_ranking(self, ranking):
        """Provide explanation for why movies are ranked highly"""
        explanations = []
        for i, movie in enumerate(ranking[:5]):  # Top 5 explanations
            explanation = f"#{i+1}: {movie['title']} (Rating: {movie['predicted_rating']:.2f})"
            explanation += f" - {movie['genres']} from {movie['year']}"
            explanations.append(explanation)
        return explanations