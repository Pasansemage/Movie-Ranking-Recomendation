import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import sys
import os

# Get project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.preprocessor import DataPreprocessor
from src.model import RecommendationModel, BaselineModel
from config.settings import *

def cross_validate_models():
    # Load data with absolute path
    data_path = os.path.join(project_root, COMBINED_DATA_PATH)
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features available: {list(df.columns)[:10]}...")  # Show first 10 columns
    
    # User activity stats
    user_counts = df['user_id'].value_counts()
    print(f"User activity stats:")
    print(f"Min ratings per user: {user_counts.min()}")
    print(f"Max ratings per user: {user_counts.max()}")
    print(f"Mean ratings per user: {user_counts.mean():.1f}")
    
    # 5-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    rf_scores = []
    xgb_scores = []
    baseline_scores = []
    
    print(f"\nRunning 5-Fold Cross Validation...")
    print(f"Training on {len(df)} samples")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(df), 1):
        print(f"Fold {fold}/5")
        
        # Split dataframes first
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        # Prepare features separately for each fold
        preprocessor = DataPreprocessor()
        X_train, y_train = preprocessor.prepare_features(train_df, is_training=True)
        X_test, y_test = preprocessor.prepare_features(test_df, is_training=False)
        
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
    # Get feature count from first fold
    sample_df = df.head(1000)
    sample_preprocessor = DataPreprocessor()
    sample_X, _ = sample_preprocessor.prepare_features(sample_df, is_training=True)
    
    print(f"Total features used: {sample_X.shape[1]}")
    
    # Count different feature types
    genre_features = len([col for col in sample_X.columns if col in ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']])
    rating_features = len([col for col in sample_X.columns if 'rating' in col])
    
    print(f"Genre binary features: {genre_features}")
    print(f"Rating-based features: {rating_features}")
    print(f"Other features: {sample_X.shape[1] - genre_features - rating_features}")
    
    return rf_scores, xgb_scores, baseline_scores

if __name__ == "__main__":
    rf_scores, xgb_scores, baseline_scores = cross_validate_models()