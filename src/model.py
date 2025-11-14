import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class RecommendationModel:
    def __init__(self, model_type='rf'):
        if model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=200, 
                max_depth=15, 
                min_samples_split=5, 
                min_samples_leaf=2, 
                random_state=42, 
                n_jobs=-1
            )
        elif model_type == 'xgb' and XGBOOST_AVAILABLE:
            self.model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        else:
            self.model = LinearRegression()
        self.model_type = model_type
        
    def train(self, X, y, sample_weight=None):
        """Train the recommendation model"""
        self.model.fit(X, y, sample_weight=sample_weight)
        
    def predict(self, X):
        """Predict ratings for given features"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        return {'mse': mse, 'mae': mae, 'rmse': np.sqrt(mse)}
    
    def save(self, filepath):
        """Save trained model"""
        joblib.dump(self.model, filepath)
        
    def load(self, filepath):
        """Load trained model"""
        self.model = joblib.load(filepath)
        return self

class BaselineModel:
    def __init__(self):
        self.user_means = {}
        self.item_means = {}
        self.global_mean = 0
        
    def train(self, df):
        """Train baseline model using user and item averages"""
        self.global_mean = df['rating'].mean()
        self.user_means = df.groupby('user_id')['rating'].mean().to_dict()
        self.item_means = df.groupby('item_id')['rating'].mean().to_dict()
        
    def predict(self, user_id, item_id):
        """Predict rating using weighted average of user and item means"""
        user_mean = self.user_means.get(user_id, self.global_mean)
        item_mean = self.item_means.get(item_id, self.global_mean)
        return (user_mean + item_mean) / 2