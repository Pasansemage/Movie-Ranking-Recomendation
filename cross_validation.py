import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from src.preprocessor import DataPreprocessor
from src.model import RecommendationModel, BaselineModel

def cross_validate_models():
    # Load data
    df = pd.read_csv('ml100k_combined.csv')
    df['genres'] = df['genres'].fillna('[]')
    
    # Calculate user activity weights
    user_counts = df['user_id'].value_counts()
    df['user_weight'] = df['user_id'].map(lambda x: 1.0 / user_counts[x])
    
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
    
    print("\nRunning 5-Fold Cross Validation...")
    
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
    
    best_model = "XGBoost" if np.mean(xgb_scores) < np.mean(rf_scores) else "Random Forest"
    improvement = abs(np.mean(rf_scores) - np.mean(xgb_scores))
    print(f"\nBest model: {best_model}")
    print(f"Improvement: {improvement:.3f} RMSE difference")
    
    return rf_scores, xgb_scores, baseline_scores

if __name__ == "__main__":
    rf_scores, xgb_scores, baseline_scores = cross_validate_models()