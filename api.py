from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
from src.preprocessor import DataPreprocessor
from src.model import RecommendationModel, BaselineModel
from src.ranker import MovieRanker
from src.collaborative_filter import GraphBasedCollaborativeFilter

app = FastAPI(title="Movie Recommendation API")

# Global variables for models
ranker = None
df = None
movie_metadata = None

class RankingRequest(BaseModel):
    user_id: int
    candidate_movies: List[int]
    use_baseline: bool = False

class MovieRanking(BaseModel):
    movie_id: int
    title: str
    predicted_rating: float
    genres: str
    year: float

@app.on_event("startup")
async def load_models():
    global ranker, df, movie_metadata
    
    print("Loading pre-trained models...")
    
    # Load data (only user data needed for inference)
    df = pd.read_csv('ml100k_combined.csv')
    df['genres'] = df['genres'].fillna('[]')
    
    # Load movie metadata
    movie_metadata = pd.read_csv('models/movie_metadata.csv')
    
    # Load pre-trained models
    preprocessor = DataPreprocessor().load('models/preprocessor.pkl')
    ml_model = RecommendationModel('rf').load('models/ml_model.pkl')
    baseline_model = joblib.load('models/baseline_model.pkl')
    
    # Create ranker with loaded models
    ranker = MovieRanker(ml_model, preprocessor, baseline_model)
    
    # Load collaborative filter if available
    try:
        collaborative_filter = joblib.load('models/collaborative_filter.pkl')
        ranker.collaborative_filter = collaborative_filter
        from src.collaborative_filter import HybridRecommender
        ranker.hybrid_recommender = HybridRecommender(collaborative_filter, ml_model, preprocessor)
        print("Collaborative filter loaded successfully!")
    except FileNotFoundError:
        print("Collaborative filter not found, using ML model only")
    
    ranker.set_movie_data(df)  # Use full dataset for similarity calculations
    
    print("Models loaded successfully!")

@app.post("/rank", response_model=List[MovieRanking])
async def rank_movies(request: RankingRequest):
    """Rank movies for a given user"""
    if ranker is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    # Validate user exists
    if request.user_id not in df['user_id'].values:
        raise HTTPException(status_code=404, detail=f"User {request.user_id} not found")
    
    # Validate movies exist
    invalid_movies = [m for m in request.candidate_movies if m not in movie_metadata['item_id'].values]
    if invalid_movies:
        raise HTTPException(status_code=404, detail=f"Movies not found: {invalid_movies}")
    
    try:
        rankings = ranker.rank_movies(
            request.user_id, 
            request.candidate_movies, 
            df, 
            use_baseline=request.use_baseline
        )
        
        return [MovieRanking(**ranking) for ranking in rankings]
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Movie Recommendation API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)