from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import re
import os
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 🔹 Initialize FastAPI
app = FastAPI()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 Initialize Firebase
try:
    firebase_credentials = os.getenv("FIREBASE_CREDENTIALS")
    if not firebase_credentials:
        print("FIREBASE_CREDENTIALS environment variable not set.")
        exit(1)
    try:
        cred_dict = json.loads(firebase_credentials)
        if "private_key" in cred_dict:
            cred_dict["private_key"] = cred_dict["private_key"].strip()
    except json.JSONDecodeError as e:
        print(f"Invalid FIREBASE_CREDENTIALS format: {e}")
        exit(1)
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    exit(1)

# 🔹 Fetch Movie Data from Firebase (Only 'playing now films' Collection)
def fetch_movies_from_firebase():
    try:
        print("Fetching movies from Firestore...")
        collection_name = 'playing now films'
        collection_ref = db.collection(collection_name)
        docs = collection_ref.limit(100).stream()
        movies = [doc.to_dict() for doc in docs]
        print(f"Fetched {len(movies)} movies from {collection_name}")
        # Print raw data for debugging
        print(f"Raw data from {collection_name}:")
        for movie in movies:
            print(movie)
        
        if not movies:
            print("No movies found in Firestore collection.")
            return pd.DataFrame()
        df = pd.DataFrame(movies)
        print(f"Total fetched movies: {len(df)}")
        return df
    except Exception as e:
        print(f"Error fetching movies: {e}")
        return pd.DataFrame()

df = fetch_movies_from_firebase()
if df.empty:
    print("No data loaded from Firestore. Exiting...")
    exit(1)

# Print available actors, categories, and directors for debugging
all_actors = set(actor for sublist in df['cast'].dropna() for actor in (sublist if isinstance(sublist, list) else []))
all_categories = set(df['category'].dropna().unique())
all_directors = set(
    crew.get('director', '') for crew in df['crew'].dropna() 
    if isinstance(crew, dict) and 'director' in crew and crew['director']
)
print("Available actors:", all_actors)
print("Available categories:", all_categories)
print("Available directors:", all_directors)

# 🔹 Clean Data
df['description'] = df['description'].fillna('')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)  # Default to 0 if missing
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce').fillna(pd.Timestamp('2000-01-01'))  # Default date if missing
# Standardize category (capitalize and strip)
df['category'] = df['category'].astype(str).str.strip().str.capitalize()
# Drop rows where category is empty
df = df[df['category'] != '']

# 🔹 TF-IDF for Similar Movie Recommendations
df['combined_features'] = (
    df['category'].fillna('') + ' ' +
    df['description'].fillna('') + ' ' +
    df['crew'].apply(lambda x: x.get('director', '') if isinstance(x, dict) else '').fillna('')
)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

# 🔹 Recommendation Functions
def recommend_by_genre(genre):
    genre = genre.lower()
    genre_mapping = {
        "action": "Action",
        "comedy": "Comedy",
        "drama": "Drama",
        "science fiction": "Science Fiction",
        "sci-fi": "Science Fiction",
        "superhero": "Superhero",
        "animation": "Animation",
        "sports": "Sports",
        "war": "War",
        "thriller": "Thriller"
    }
    standard_genre = genre_mapping.get(genre, genre.capitalize())
    matches = df[df['category'].str.lower() == standard_genre.lower()]  # Exact match
    return matches.sort_values(by=['rating', 'release_date'], ascending=[False, False]).head(5)

def recommend_by_actor(actor):
    return df[df['cast'].apply(lambda x: actor.lower() in [a.lower() for a in x] if isinstance(x, list) else False)]\
        .sort_values(by=['rating', 'release_date'], ascending=[False, False]).head(5)

def recommend_by_director(director):
    return df[df['crew'].apply(lambda x: director.lower() in x.get('director', '').lower() if isinstance(x, dict) else False)]\
        .sort_values(by=['rating', 'release_date'], ascending=[False, False]).head(5)

def recommend_similar_movies(movie_title):
    if movie_title not in df['name'].values:
        return pd.DataFrame()
    idx = df[df['name'] == movie_title].index[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    indices = cosine_sim.argsort()[-6:-1][::-1]
    return df.iloc[indices].sort_values(by=['rating', 'release_date'], ascending=[False, False])

def search_by_movie_name(movie_name):
    matches = df[df['name'].str.lower() == movie_name.lower()]  # Exact match
    return matches.sort_values(by=['rating', 'release_date'], ascending=[False, False]).head(5)

# 🔹 Greetings, Help, and Thank You Handler
def handle_greetings_or_help_or_thanks(user_input):
    greetings = ["hello", "hi", "hey", "hay", "good morning", "good evening", "good afternoon"]
    help_requests = ["help", "can you help", "what can you do", "tell me about movies", "recommend a movie"]
    thank_you_keywords = ["thank you", "thanks", "thx", "thank u"]
    user_input_lower = user_input.lower()
    
    for thank in thank_you_keywords:
        if thank in user_input_lower and not any(word in user_input_lower for word in greetings + help_requests):
            return "You're welcome! Let me know if you need more help."
    
    for greeting in greetings + help_requests:
        if greeting in user_input_lower:
            return "Hello! How can I assist you today? Feel free to ask for movie recommendations or anything else."
    
    return None

# 🔹 Check for Arabic Text
def contains_arabic(text):
    return bool(re.search(r'[؀-ۿ]', text))

# 🔹 Plain Text Formatter
def format_recommendations_json(recommendations):
    if not recommendations.empty:
        formatted_recommendations = []
        for _, row in recommendations.iterrows():
            try:
                year = row['release_date'].year if pd.notnull(row['release_date']) else 'Unknown'
            except AttributeError:
                year = 'Unknown'
            try:
                cast_list = row['cast']
                if isinstance(cast_list, str):
                    cast_list = ast.literal_eval(cast_list)
            except Exception:
                cast_list = []
            try:
                director = row['crew']['director'] if isinstance(row['crew'], dict) and 'director' in row['crew'] else 'Unknown'
            except Exception:
                director = 'Unknown'
            formatted_recommendations.append({
                "title": row['name'],
                "genre": row['category'],
                "imdb_score": row['rating'],
                "release_year": year,
                "cast": cast_list,
                "director": director
            })
        return formatted_recommendations
    return []

# 🔹 Classify Input
def classify_input(user_input):
    user_input_lower = user_input.lower().strip()
    print(f"Classifying input: {user_input_lower}")
    
    # Check for greetings, help, or thank you first
    if handle_greetings_or_help_or_thanks(user_input_lower):
        return "greeting_help_or_thanks", None
    
    # Check for genre (exact match with keywords) first
    genre_keywords = ["action", "comedy", "drama", "science fiction", "sci-fi", "superhero", "animation", "sports", "war", "thriller"]
    matched_genre = None
    for genre in genre_keywords:
        if user_input_lower == genre or user_input_lower == f"{genre} movie" or user_input_lower == f"{genre} movies":
            if any(genre == g.lower() for g in all_categories):
                matched_genre = genre
                break
    if matched_genre:
        print(f"Matched genre: {matched_genre}")
        return "genre", matched_genre
    
    # Check for movie name (exact match)
    for title in df['name'].dropna().unique():
        if user_input_lower == title.lower():  # Exact match
            print(f"Matched movie name: {title}")
            return "movie_name", title
    
    # Check for actor within the sentence
    all_actors = set(actor for sublist in df['cast'].dropna() for actor in (sublist if isinstance(sublist, list) else []))
    found_actor = None
    for actor in all_actors:
        if actor.lower() in user_input_lower:  # Check if actor name appears in the input
            found_actor = actor
            break
    if found_actor:
        print(f"Matched actor: {found_actor}")
        return "actor", found_actor
    
    # Check for director within the sentence
    all_directors = set(
        crew.get('director', '') for crew in df['crew'].dropna() 
        if isinstance(crew, dict) and 'director' in crew and crew['director']
    )
    found_director = None
    for director in all_directors:
        if director.lower() in user_input_lower:  # Check if director name appears in the input
            found_director = director
            break
    if found_director:
        print(f"Matched director: {found_director}")
        return "director", found_director
    
    # Default to unknown
    print("No match found")
    return "unknown", user_input

# 🔹 Final Bot Response
def get_bot_response(user_input):
    # Reject any input containing Arabic characters
    if contains_arabic(user_input):
        return {
            "bot": "Please use English only for communication.",
            "recommendations": []
        }
    
    # Handle empty input
    if not user_input or user_input.strip() == "":
        return {
            "bot": "It looks like you didn't enter anything. Please try a genre, actor, or movie name!",
            "recommendations": []
        }
    
    # Classify the input
    label, value = classify_input(user_input)
    user_input_lower = user_input.lower()

    # Handle greetings, help, or thank you
    if label == "greeting_help_or_thanks":
        response = handle_greetings_or_help_or_thanks(user_input_lower)
        return {
            "bot": response,
            "recommendations": []
        }

    # Handle movie name search
    if label == "movie_name":
        recs = search_by_movie_name(value)
        if not recs.empty:
            formatted_recs = format_recommendations_json(recs)
            return {
                "bot": f"Movies found for '{value}'",
                "recommendations": formatted_recs
            }
        return {
            "bot": f"No movies found for '{value}'",
            "recommendations": []
        }

    # Handle genre-based recommendations
    if label == "genre":
        recs = recommend_by_genre(value)
        if not recs.empty:
            formatted_recs = format_recommendations_json(recs)
            return {
                "bot": f"Movies found for '{value.capitalize()}'",
                "recommendations": formatted_recs
            }
        return {
            "bot": f"No movies found for '{value.capitalize()}'",
            "recommendations": []
        }

    # Handle actor-based recommendations
    if label == "actor":
        recs = recommend_by_actor(value)
        if not recs.empty:
            formatted_recs = format_recommendations_json(recs)
            return {
                "bot": f"Movies found for '{value}'",
                "recommendations": formatted_recs
            }
        return {
            "bot": f"No movies found for actor '{value}'",
            "recommendations": []
        }

    # Handle director-based recommendations
    if label == "director":
        recs = recommend_by_director(value)
        if not recs.empty:
            formatted_recs = format_recommendations_json(recs)
            return {
                "bot": f"Movies found for director '{value}'",
                "recommendations": formatted_recs
            }
        return {
            "bot": f"No movies found for director '{value}'",
            "recommendations": []
        }

    # Default response for unrecognized input
    return {
        "bot": f"No results found for '{user_input}'. Try a movie name, genre, actor, or director.",
        "recommendations": []
    }

# 🔹 Root Endpoint
@app.get("/")
async def root():
    return {
        "response": {
            "bot": "Welcome to the Movie Recommendation API! Use POST /recommend to get recommendations.",
            "recommendations": []
        }
    }

# 🔹 API Endpoint
@app.post("/recommend")
async def recommend(request: dict):
    user_input = request.get("message", "")
    if not user_input:
        return JSONResponse(content={
            "response": {
                "bot": "Missing message",
                "recommendations": []
            }
        }, status_code=400)
    response = get_bot_response(user_input)
    return {"response": response}

# 🔹 Endpoint to Get All Movies
@app.get("/movies")
async def get_all_movies():
    formatted_movies = format_recommendations_json(df)
    return {
        "response": {
            "bot": "Here are all the movies in the database.",
            "recommendations": formatted_movies
        }
    }

# 🔹 Run Server
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)