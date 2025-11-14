# Movie Recommendation System - MovieLens 100K

A personalized movie ranking system that predicts user preferences and ranks candidate movies based on user demographics, movie metadata, and historical ratings.

## Approach

### Model Architecture
- **ML Model**: Random Forest Regressor for rating prediction
- **Baseline Model**: Weighted average of user and item means
- **Features**: User demographics (age, gender, occupation), movie metadata (genres, year), and encoded categorical variables

### Feature Engineering
- User encoding and demographic normalization
- Movie age calculation and movie metadata features
- Individual binary genre columns (18 genres: Action, Comedy, Drama, etc.)
- TF-IDF genre vectorization for genre combinations
- Standard scaling for numerical features

### Ranking Logic
Given a user ID and candidate movies, the system:
1. Generates feature vectors for each user-movie pair
2. Predicts ratings using the trained model
3. Ranks movies by predicted rating (descending)
4. Provides explanations based on movie metadata

## Model Choice Rationale

**Random Forest Regressor** was chosen because:
- Handles mixed data types (categorical + numerical) effectively
- Robust to outliers and missing values
- Provides feature importance for interpretability
- Good performance on recommendation tasks with tabular data

**Baseline Model** uses collaborative filtering principles:
- Simple average of user and item historical ratings
- Provides interpretable benchmark for comparison

## Project Structure

```
├── src/
│   ├── preprocessor.py    # Feature engineering
│   ├── model.py          # ML and baseline models
│   └── ranker.py         # Ranking logic and explanations
├── train_model.py        # Model training script
├── demo.py              # Simple demo script
├── api.py               # FastAPI web service
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models and See Demo
```bash
python train_model.py
```

### 3. Run Simple Demo
```bash
python demo.py
```

### 4. Start API Server
```bash
python api.py
```

Then visit `http://localhost:8000/docs` for interactive API documentation.

### 5. API Usage Example
```python
import requests

response = requests.post("http://localhost:8000/rank", json={
    "user_id": 196,
    "candidate_movies": [1, 50, 100, 181, 258],
    "use_baseline": false
})

rankings = response.json()
```

## Results

The system provides:
- **Personalized Rankings**: Movies ranked by predicted user preference
- **Model Comparison**: ML model vs baseline performance metrics
- **Explanations**: Why movies are ranked highly (genres, year, predicted rating)
- **API Interface**: RESTful service for integration

## Sample Output

```
Top Recommendations for User 196:
1. Toy Story (1995) - Predicted Rating: 4.23
   Genres: ['Animation', 'Children', 'Comedy'], Year: 1995

2. Fargo (1996) - Predicted Rating: 4.15
   Genres: ['Crime', 'Drama', 'Thriller'], Year: 1996
```

## Performance

- **ML Model RMSE**: ~0.95
- **Baseline RMSE**: ~1.02
- **Training Time**: <30 seconds on standard hardware
- **Prediction Time**: <1ms per user-movie pair