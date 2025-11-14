# Configuration settings for the recommendation system

# Data paths
DATA_DIR = "data"
MODELS_DIR = "models"
ML100K_DIR = f"{DATA_DIR}/ml-100k"
COMBINED_DATA_PATH = f"{DATA_DIR}/ml100k_combined.csv"

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# XGBoost parameters
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'random_state': RANDOM_STATE,
    'verbosity': 0
}

# Random Forest parameters
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Rating constraints
MIN_RATING = 1.0
MAX_RATING = 5.0

# Genre names
GENRE_NAMES = [
    'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000