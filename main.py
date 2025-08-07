import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Depends
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")
print("Database URL", DATABASE_URL)
API_KEY = os.environ.get("API_KEY")

if not API_KEY:
    raise ValueError("Not found")

def verify_api_key(
    api_key: str) -> bool:
    return  api_key == API_KEY

def authenticate_user(    
    api_key: Optional[str] = Header(None, alias="APIKey")
):
    
    
    if not verify_api_key(api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid user ID or API key"
        )
    
    return {"api_key": api_key}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this based on your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = None
tfidf_matrix = None
cosine_sim = None
indices = None

def fetch_products():
    engine = create_engine(DATABASE_URL)
    query = "SELECT * FROM product;"
    df = pd.read_sql(query, engine)
    engine.dispose()
    return df

def initialize_recommendation_model():
    global df, tfidf_matrix, cosine_sim, indices

    df = fetch_products()
    
    text_columns = ['name', 'category', 'description', 'highlights', 'tags']
    for col in text_columns:
        df[col] = df[col].fillna('')

    df['combined_text_features'] = df[text_columns].astype(str).apply(lambda row: ' '.join(row), axis=1)

    tfidf = TfidfVectorizer(stop_words='english', min_df=1, max_df=0.8)
    tfidf_matrix = tfidf.fit_transform(df['combined_text_features'])
    
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    df = df.drop_duplicates(subset=['id'])
    indices = pd.Series(df.index, index=df['id'])

initialize_recommendation_model()

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/recommend")
def recommend(
    productid: str, 
    top_n: int = 5,
    auth: dict = Depends(authenticate_user)
):
    if productid not in indices:
        return {"error": "Product not found"}

    idx = indices[productid]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    results = []
    for i, _ in sim_scores:
        row = df.iloc[i]
        
        results.append({
            "productid": row["id"],
            "name": row["name"],
            "category": row.get("category", "N/A"),
            "listingPrice": row["listingPrice"],
        })
    
    return {        
        "recommendations": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=True)
    # uvicorn main:app --reload --port 8000
