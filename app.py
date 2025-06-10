# --------------------------------------------------------------------------
# app.py (v2.6 - Genre-Based Similarity)
# --------------------------------------------------------------------------

import os
import json
import re
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timedelta
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

# --- 1. إعدادات البيئة والتطبيق ---
load_dotenv()
app = FastAPI(
    title="Movie Recommendation Chatbot API",
    description="An API for a chatbot that recommends movies based on various criteria.",
    version="2.6.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. نماذج Pydantic ---
class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's input message to the chatbot.")
class MovieRecommendation(BaseModel):
    title: str
    genre: str
    imdb_score: float
    release_year: int
    cast: List[str]
    director: str
class ChatResponse(BaseModel):
    bot: str
    recommendations: List[MovieRecommendation]
class RootResponse(BaseModel):
    message: str
    documentation_url: str

# --- 3. آلية التخزين المؤقت (Caching) ---
class AppState:
    df: pd.DataFrame = None
    all_titles_lower: set = set()
    all_actors_lower: set = set()
    all_directors_lower: set = set()
    last_fetch_time: Optional[datetime] = None
    CACHE_DURATION: timedelta = timedelta(hours=1)
app_state = AppState()

# --- 4. الاتصال بـ Firebase ---
try:
    firebase_credentials = os.getenv("FIREBASE_CREDENTIALS")
    if not firebase_credentials:
        raise ValueError("FIREBASE_CREDENTIALS environment variable not set.")
    cred_dict = json.loads(firebase_credentials)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(cred_dict))
    db = firestore.client()
    print("✅ Firebase initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing Firebase: {e}")

def fetch_movies_from_firebase() -> pd.DataFrame:
    try:
        print("Fetching movies from Firestore...")
        collection_ref = db.collection('playing now films')
        docs = collection_ref.limit(200).stream()
        movies = [doc.to_dict() for doc in docs]
        if not movies:
            print("⚠️ No movies found in Firestore collection.")
            return pd.DataFrame()
        print(f"✅ Fetched {len(movies)} movies from Firestore.")
        return pd.DataFrame(movies)
    except Exception as e:
        print(f"❌ Error fetching movies: {e}")
        return pd.DataFrame()

# --- 5. معالجة البيانات ---
def load_and_process_data():
    global app_state
    now = datetime.now()
    if app_state.df is not None and app_state.last_fetch_time and (now - app_state.last_fetch_time) < app_state.CACHE_DURATION:
        print("Using cached data.")
        return
    print("Loading and processing new data...")
    df = fetch_movies_from_firebase()
    if df.empty:
        print("Stopping process as no data was loaded.")
        return
    df['description'] = df['description'].fillna('')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0.0)
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['category'] = df['category'].astype(str).str.strip().str.capitalize()
    df.dropna(subset=['release_date', 'category'], inplace=True)
    df = df[df['category'] != '']
    app_state.df = df
    app_state.all_titles_lower = {str(title).lower() for title in df['name'].dropna()}
    app_state.all_actors_lower = {str(actor).lower().strip() for sublist in df['cast'].dropna() if isinstance(sublist, list) for actor in sublist}
    app_state.all_directors_lower = {str(crew.get('director', '')).lower().strip() for crew in df['crew'].dropna() if isinstance(crew, dict) and crew.get('director')}
    app_state.last_fetch_time = now
    print("✅ Data loaded and cached successfully.")

# --- 6. دوال منطق الشات بوت والتوصية ---

def contains_arabic(text: str) -> bool:
    return bool(re.search(r'[\u0600-\u06FF]', text))

def recommend_by_actor(actor_name: str) -> pd.DataFrame:
    df = app_state.df
    actor_lower = actor_name.lower()
    mask = df['cast'].apply(
        lambda cast_list: any(actor_lower == str(actor).lower().strip() for actor in cast_list) if isinstance(cast_list, list) else False
    )
    return df[mask].sort_values(by=['rating', 'release_date'], ascending=[False, False])

def recommend_by_director(director_name: str) -> pd.DataFrame:
    df = app_state.df
    director_lower = director_name.lower()
    mask = df['crew'].apply(
        lambda crew_dict: director_lower == str(crew_dict.get('director', '')).lower().strip() if isinstance(crew_dict, dict) else False
    )
    return df[mask].sort_values(by=['rating', 'release_date'], ascending=[False, False])

def recommend_by_genre(genre: str) -> pd.DataFrame:
    df = app_state.df
    genre_mapping = { "sci-fi": "Science Fiction", "science fiction": "Science Fiction"}
    standard_genre = genre_mapping.get(genre.lower(), genre.capitalize())
    matches = df[df['category'].str.lower() == standard_genre.lower()]
    return matches.sort_values(by=['rating', 'release_date'], ascending=[False, False]).head(5)

# (## تعديل جذري هنا ##): دالة جديدة للتوصية بناء على تصنيف فيلم آخر
def recommend_by_same_genre_as_movie(movie_title: str) -> pd.DataFrame:
    df = app_state.df
    movie_title_lower = movie_title.lower()
    
    # 1. البحث عن الفيلم المستهدف
    target_movie = df[df['name'].str.lower() == movie_title_lower]
    
    # التأكد من العثور على الفيلم
    if target_movie.empty:
        return pd.DataFrame() # إرجاع جدول فارغ إذا لم يتم العثور على الفيلم
    
    # 2. استخراج التصنيف
    target_category = target_movie['category'].iloc[0]
    
    # 3. الحصول على أفلام من نفس التصنيف
    recommendations = recommend_by_genre(target_category)
    
    # 4. إزالة الفيلم الأصلي من قائمة التوصيات
    final_recommendations = recommendations[recommendations['name'].str.lower() != movie_title_lower]
    
    return final_recommendations


def format_recommendations(recs_df: pd.DataFrame) -> List[dict]:
    if recs_df.empty: return []
    recs_df = recs_df.copy()
    recs_df['release_year'] = recs_df['release_date'].dt.year
    recs_df['director'] = recs_df['crew'].apply(lambda x: x.get('director', 'Unknown') if isinstance(x, dict) else 'Unknown')
    recs_df['cast'] = recs_df['cast'].apply(lambda x: x if isinstance(x, list) else [])
    recs_df.rename(columns={'name': 'title', 'rating': 'imdb_score', 'category': 'genre'}, inplace=True)
    output_columns = ['title', 'genre', 'imdb_score', 'release_year', 'cast', 'director']
    recs_df_filtered = recs_df[[col for col in output_columns if col in recs_df.columns]]
    return recs_df_filtered.to_dict(orient='records')

def classify_input(user_input: str) -> Tuple[str, Optional[str]]:
    user_input_lower = user_input.lower().strip()
    
    # إضافة كلمات مفتاحية جديدة للبحث عن أفلام مشابهة
    similarity_triggers = {"similar to", "like"}
    
    # التحقق من نية البحث عن فيلم مشابه أولاً
    if any(trigger in user_input_lower for trigger in similarity_triggers):
        # محاولة استخلاص اسم الفيلم بعد الكلمة المفتاحية
        for trigger in similarity_triggers:
            if trigger in user_input_lower:
                # تقسيم الجملة عند الكلمة المفتاحية وأخذ ما بعدها
                potential_title = user_input_lower.split(trigger, 1)[1].strip()
                # البحث عن هذا العنوان في قائمة الأفلام المعروفة
                for title in app_state.all_titles_lower:
                    if title == potential_title:
                        original_title = next((t for t in app_state.df['name'] if t.lower() == title), title)
                        return "similar_movie_search", original_title

    actor_triggers = {"actor", "starring", "acted in", "movies with", "by actor"}
    if any(trigger in user_input_lower for trigger in actor_triggers):
        found_actors = [actor for actor in app_state.all_actors_lower if re.search(r'\b' + re.escape(actor) + r'\b', user_input_lower)]
        if found_actors:
            longest_match = max(found_actors, key=len)
            return "actor_search", longest_match.title()
        else:
            return "actor_not_found", user_input

    director_triggers = {"directed by", "director", "by director"}
    if any(trigger in user_input_lower for trigger in director_triggers):
        found_directors = [director for director in app_state.all_directors_lower if re.search(r'\b' + re.escape(director) + r'\b', user_input_lower)]
        if found_directors:
            longest_match = max(found_directors, key=len)
            return "director_search", longest_match.title()
        else:
            return "director_not_found", user_input
            
    genre_keywords = {"action", "comedy", "drama", "science fiction", "sci-fi", "superhero", "animation", "sports", "war", "thriller"}
    for genre in genre_keywords:
        if re.search(r'\b' + re.escape(genre) + r'\b', user_input_lower):
            return "genre", genre

    thank_yous = {"thank you", "thanks", "thx", "thank u"}
    if any(word in user_input_lower for word in thank_yous):
        return "thank_you", None
        
    greetings = {"hello", "hi", "hey"}
    help_requests = {"help", "can you help", "what can you do", "recommend a movie"}
    if any(word in user_input_lower for word in greetings | help_requests):
        return "greeting_or_help", None
        
    return "unknown", user_input

def get_bot_response(user_input: str) -> dict:
    if contains_arabic(user_input):
        return { "bot": "Sorry, I only understand English. Please use English for communication.", "recommendations": [] }
    if not user_input or not user_input.strip():
        return {"bot": "Please provide a movie title, genre, actor, or director.", "recommendations": []}

    label, value = classify_input(user_input)
    bot_message = ""
    recs_df = pd.DataFrame()

    if label == "greeting_or_help":
        bot_message = "Hello! I can recommend movies."
    elif label == "thank_you":
        bot_message = "You're welcome!"
    elif label == "genre":
        bot_message = f"Looking for {value} movies? Here are some great ones:"
        recs_df = recommend_by_genre(value)
    elif label == "actor_search":
        bot_message = f"You got it! Here are movies starring '{value}':"
        recs_df = recommend_by_actor(value)
    elif label == "director_search":
        bot_message = f"Of course! Here are movies directed by '{value}':"
        recs_df = recommend_by_director(value)
    # (## تعديل هنا ##): إضافة حالة جديدة للبحث عن أفلام مشابهة
    elif label == "similar_movie_search":
        bot_message = f"If you liked '{value}', you might also enjoy these movies from the same genre:"
        recs_df = recommend_by_same_genre_as_movie(value)
    elif label == "actor_not_found":
        bot_message = "Sorry, I couldn't find a matching actor in the database. Please check the spelling."
    elif label == "director_not_found":
        bot_message = "Sorry, I couldn't find a matching director in the database. Please check the spelling."
    else: # label == "unknown"
        bot_message = f"I see you're asking about '{user_input}'. To give you the best results, could you please be more specific? For example, try 'movies starring {user_input}'."
        recs_df = pd.DataFrame()

    formatted_recs = format_recommendations(recs_df)
    
    if not formatted_recs and label in ["actor_search", "director_search", "similar_movie_search"]:
        bot_message = f"I found '{value}' in my database, but it seems there are no available movies for this request at the moment."
    elif not formatted_recs and label == "genre":
        bot_message = f"Sorry, no movies found for the genre '{value}'. Please try something else."
            
    return {"bot": bot_message, "recommendations": formatted_recs}


# --- 7. نقاط النهاية (API Endpoints) ---
@app.on_event("startup")
def on_startup():
    load_and_process_data()
@app.get("/", response_model=RootResponse, tags=["General"])
def root():
    return { "message": "Welcome to the Movie Recommendation API!", "documentation_url": "/docs" }
@app.post("/recommend", response_model=ChatResponse, tags=["Chatbot"])
def recommend(request: ChatRequest):
    if app_state.df is None:
        raise HTTPException(status_code=503, detail="Service Unavailable: Movie data is not loaded. Please try again later.")
    response_data = get_bot_response(request.message)
    return response_data

# --- 8. تشغيل السيرفر ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)