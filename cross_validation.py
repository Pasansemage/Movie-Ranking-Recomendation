import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from src.preprocessor import DataPreprocessor
from src.model import RecommendationModel, BaselineModel

def cross_validate_models():
    # Load data
    df = pd.read_csv('ml100k_combined.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features available: {list(df.columns)[:10]}...")  # Show first 10 columns
    
    # User activity stats
    user_counts = df['user_id'].value_counts()
    print(f"User activity stats:")
    print(f"Min ratings per user: {user_counts.min()}")
    print(f"Max ratings per user: {user_counts.max()}")
    print(f"Mean ratings per user: {user_counts.mean():.1f}")
    
    # Prepare features
    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_features(df, is_training=True)
    
    # 5-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    rf_scores = []
    xgb_scores = []
    baseline_scores = []
    
    print(f"\nRunning 5-Fold Cross Validation with {X.shape[1]} features...")
    print(f"Training on {len(df)} samples")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        print(f"Fold {fold}/5")
        
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Random Forest Model
        rf_model = RecommendationModel('rf')
        rf_model.train(X_train, y_train)
        rf_metrics = rf_model.evaluate(X_test, y_test)
        rf_scores.append(rf_metrics['rmse'])
        
        # XGBoost Model
        xgb_model = RecommendationModel('xgb')
        xgb_model.train(X_train, y_train)
        xgb_metrics = xgb_model.evaluate(X_test, y_test)
        xgb_scores.append(xgb_metrics['rmse'])
        
        # Baseline Model
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        baseline_model = BaselineModel()
        baseline_model.train(train_df)
        
        baseline_preds = [baseline_model.predict(row['user_id'], row['item_id']) 
                         for _, row in test_df.iterrows()]
        baseline_rmse = np.sqrt(np.mean((y_test - baseline_preds) ** 2))
        baseline_scores.append(baseline_rmse)
        
        print(f"  RF RMSE: {rf_metrics['rmse']:.3f}, XGB RMSE: {xgb_metrics['rmse']:.3f}, Baseline RMSE: {baseline_rmse:.3f}")
    
    # Results
    print(f"\n5-Fold Cross Validation Results:")
    print(f"Random Forest - Mean RMSE: {np.mean(rf_scores):.3f} ± {np.std(rf_scores):.3f}")
    print(f"XGBoost - Mean RMSE: {np.mean(xgb_scores):.3f} ± {np.std(xgb_scores):.3f}")
    print(f"Baseline - Mean RMSE: {np.mean(baseline_scores):.3f} ± {np.std(baseline_scores):.3f}")
    
    # Model comparison
    best_model = "XGBoost" if np.mean(xgb_scores) < np.mean(rf_scores) else "Random Forest"
    best_score = min(np.mean(rf_scores), np.mean(xgb_scores))
    baseline_mean = np.mean(baseline_scores)
    
    print(f"\nBest ML model: {best_model} (RMSE: {best_score:.3f})")
    print(f"Improvement over baseline: {baseline_mean - best_score:.3f} RMSE")
    print(f"Relative improvement: {((baseline_mean - best_score) / baseline_mean * 100):.1f}%")
    
    # Feature importance analysis for best model
    print(f"\nFeature Analysis:")
    print(f"Total features used: {X.shape[1]}")
    
    # Count different feature types
    genre_features = len([col for col in X.columns if any(genre.lower() in col.lower() for genre in ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])])
    tfidf_features = len([col for col in X.columns if col.startswith('tfidf_')])
    rating_features = len([col for col in X.columns if 'rating' in col])
    
    print(f"Genre-related features: {genre_features}")
    print(f"TF-IDF features: {tfidf_features}")
    print(f"Rating-based features: {rating_features}")
    print(f"Other features: {X.shape[1] - genre_features - tfidf_features}")
    
    return rf_scores, xgb_scores, baseline_scores

if __name__ == "__main__":
    rf_scores, xgb_scores, baseline_scores = cross_validate_models()