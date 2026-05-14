import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import ast
import urllib.parse
import kagglehub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz

# --- CONFIG & STATE ---
st.set_page_config(page_title="CineVerse AI — Antigravity Edition", page_icon="🌌", layout="wide", initial_sidebar_state="expanded")

if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = 'dark'
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = None
if 'current_mood' not in st.session_state:
    st.session_state.current_mood = None
if 'active_section' not in st.session_state:
    st.session_state.active_section = 'Home'

TMDB_API_KEY = "YOUR_API_KEY_HERE"   # ← ADD YOUR API KEY HERE

# Handle Query Params for 'More Like This' clicks
if hasattr(st, 'query_params') and 'movie' in st.query_params:
    movie_param = st.query_params['movie']
    if isinstance(movie_param, list): movie_param = movie_param[0]
    st.session_state.selected_movie = movie_param
    st.session_state.active_section = 'Recommended'
    # Clear query param so it doesn't persist
    try:
        st.query_params.clear()
    except:
        pass

# --- CSS INJECTION ---
def inject_css():
    theme = st.session_state.theme_mode
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600&family=Rajdhani:wght@400;600&display=swap');

    :root {{
        --void: {'#04050a' if theme == 'dark' else '#faf7f2'};
        --nebula-1: {'#0d0221' if theme == 'dark' else '#f0eef5'};
        --nebula-2: {'#0a1628' if theme == 'dark' else '#e0e5ec'};
        --plasma: #e94560;
        --aurora: #00d4ff;
        --gold: #ffd700;
        --glass: {'rgba(255,255,255,0.04)' if theme == 'dark' else 'rgba(0,0,0,0.04)'};
        --glass-border: {'rgba(255,255,255,0.08)' if theme == 'dark' else 'rgba(0,0,0,0.08)'};
        --text: {'#ffffff' if theme == 'dark' else '#1a1a2e'};
        --card-bg: {'rgba(13, 2, 33, 0.7)' if theme == 'dark' else 'rgba(255, 255, 255, 0.7)'};
    }}

    .stApp {{
        background-color: var(--void);
        color: var(--text);
        font-family: 'Exo 2', sans-serif;
        background-image: 
            radial-gradient(circle at 15% 50%, rgba(233, 69, 96, 0.05), transparent 25%),
            radial-gradient(circle at 85% 30%, rgba(0, 212, 255, 0.05), transparent 25%);
        background-attachment: fixed;
    }}
    
    h1, h2, h3, h4, h5, h6, .orbitron {{
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 1px;
        color: var(--text);
    }}

    /* ANTIGRAVITY CARD PHYSICS */
    .movie-card {{
        background: var(--card-bg);
        backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        overflow: hidden;
        position: relative;
        padding: 0;
        color: var(--text);
        text-decoration: none;
        display: flex;
        flex-direction: column;
        animation: antigravity-float 6s ease-in-out infinite;
        animation-delay: calc(var(--card-index, 0) * 0.3s);
        transform-style: preserve-3d;
        transition: transform 0.4s cubic-bezier(0.23, 1, 0.32, 1),
                    box-shadow 0.4s ease;
        height: 380px;
        cursor: pointer;
    }}

    @keyframes antigravity-float {{
        0%   {{ transform: translateY(0px)   rotate(0deg);    }}
        25%  {{ transform: translateY(-8px)  rotate(0.5deg);  }}
        50%  {{ transform: translateY(-14px) rotate(-0.5deg); }}
        75%  {{ transform: translateY(-6px)  rotate(0.3deg);  }}
        100% {{ transform: translateY(0px)   rotate(0deg);    }}
    }}

    .movie-card:hover {{
        animation-play-state: paused;
        transform: translateY(-20px) rotateX(8deg) rotateY(-5deg) scale(1.06);
        box-shadow:
            0 40px 80px rgba(233, 69, 96, 0.4),
            0 0 60px rgba(0, 212, 255, 0.15),
            inset 0 1px 0 rgba(255,255,255,0.1);
        z-index: 100;
    }}

    .movie-card::before {{
        content: '';
        position: absolute;
        inset: -2px;
        border-radius: inherit;
        background: linear-gradient(135deg, rgba(233,69,96,0.6), rgba(0,212,255,0.6));
        opacity: 0;
        transition: opacity 0.3s ease;
        z-index: -1;
    }}
    .movie-card:hover::before {{ opacity: 1; }}

    .poster-container {{
        position: relative;
        width: 100%;
        height: 60%;
        overflow: hidden;
    }}
    
    .poster-img {{
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.5s ease;
    }}
    
    .movie-card:hover .poster-img {{ transform: scale(1.1); }}
    
    .card-content {{
        padding: 15px;
        flex-grow: 1;
        display: flex;
        flex-direction: column;
        background: {'linear-gradient(to top, rgba(0,0,0,0.9), rgba(0,0,0,0.2))' if theme == 'dark' else 'linear-gradient(to top, rgba(255,255,255,1), rgba(255,255,255,0.8))'};
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 55%;
    }}

    .movie-title {{
        font-family: 'Orbitron', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        margin: 0 0 5px 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}

    .movie-meta {{
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.85rem;
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
        color: var(--text);
    }}
    
    .rating {{ color: var(--gold); font-weight: 600; }}

    .genre-chips {{
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
        margin-bottom: 5px;
    }}

    .genre-chip {{
        background: var(--glass);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 2px 8px;
        font-size: 0.65rem;
        transition: all 0.2s ease;
    }}
    
    .movie-card:hover .genre-chip {{
        background: var(--plasma);
        border-color: var(--plasma);
        color: white;
    }}

    .movie-overview {{
        font-size: 0.75rem;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        margin-bottom: 10px;
        opacity: 0.8;
    }}
    
    .card-btn {{
        background: transparent;
        border: 1px solid var(--aurora);
        color: var(--aurora);
        padding: 5px 0;
        text-align: center;
        border-radius: 8px;
        font-family: 'Orbitron', sans-serif;
        font-size: 0.75rem;
        text-transform: uppercase;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-top: auto;
    }}
    
    .card-btn:hover {{
        background: var(--aurora);
        color: #000;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.5);
        transform: translateX(5px);
    }}

    /* MOOD BOARD */
    .mood-board {{
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        justify-content: center;
        padding: 20px 0;
    }}
    
    .mood-bubble {{
        background: var(--glass);
        border: 1px solid var(--glass-border);
        border-radius: 50px;
        padding: 10px 20px;
        font-size: 1.2rem;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        animation: float 4s ease-in-out infinite;
        color: var(--text);
    }}
    
    .mood-bubble:nth-child(even) {{ animation-delay: 1s; }}
    .mood-bubble:nth-child(3n) {{ animation-delay: 2s; }}
    
    .mood-bubble:hover {{
        transform: scale(1.1) translateY(-10px);
        background: rgba(233, 69, 96, 0.2);
        box-shadow: 0 10px 20px rgba(0,0,0,0.3), 0 0 15px var(--plasma);
        border-color: var(--plasma);
    }}
    
    /* BLACK HOLE LOADER */
    .black-hole-loader {{
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: radial-gradient(circle, #000 20%, transparent 60%);
        box-shadow: 0 0 30px #e94560, 0 0 60px #00d4ff;
        animation: spin 2s linear infinite;
        margin: 40px auto;
    }}
    
    @keyframes spin {{
        100% {{ transform: rotate(360deg) scale(1.1); }}
    }}
    
    .hero-banner {{
        border-radius: 20px;
        overflow: hidden;
        border: 1px solid var(--glass-border);
        background: linear-gradient(135deg, rgba(233,69,96,0.2), rgba(0,212,255,0.2));
        padding: 40px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.5);
    }}
    
    /* Section Reveal */
    .reveal {{
        animation: float-up 0.8s ease-out forwards;
    }}
    @keyframes float-up {{
        0% {{ transform: translateY(60px); opacity: 0; }}
        100% {{ transform: translateY(0); opacity: 1; }}
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- DATA HANDLING ---
@st.cache_data(show_spinner=False)
def load_data():
    import kagglehub
    import os
    import pandas as pd

    # Download latest version
    path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
    print("Path to dataset files:", path)

    files = os.listdir(path)
    print("Files found:", files)

    csv_files = [os.path.join(path, f) for f in files if f.endswith('.csv')]

    movies_path  = [f for f in csv_files if "movies"  in f][0]
    credits_path = [f for f in csv_files if "credits" in f][0]

    movies  = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    # Merge on title
    credits.rename(columns={"movie_id": "id"}, errors="ignore")
    movies  = movies.merge(credits, on="title", how="left")

    return movies, credits

@st.cache_data(show_spinner=False)
def preprocess_features(df):
    def get_list(x, limit=None):
        if pd.isna(x): return []
        try:
            lst = ast.literal_eval(x)
            res = [i['name'].replace(" ", "").lower() for i in lst]
            return res[:limit] if limit else res
        except: return []

    def get_director(x):
        if pd.isna(x): return []
        try:
            lst = ast.literal_eval(x)
            return [i['name'].replace(" ", "").lower() for i in lst if i['job'] == 'Director']
        except: return []

    df['genres_list'] = df['genres'].apply(get_list)
    df['keywords_list'] = df['keywords'].apply(get_list)
    df['cast_list'] = df['cast'].apply(lambda x: get_list(x, limit=4))
    df['director_list'] = df['crew'].apply(get_director)
    
    df['overview'] = df['overview'].fillna('')
    df['tagline'] = df['tagline'].fillna('')
    
    df['tags'] = df['overview'] + " " + df['tagline'] + " " + \
                 df['genres_list'].apply(lambda x: " ".join(x) * 2) + " " + \
                 df['keywords_list'].apply(lambda x: " ".join(x) * 2) + " " + \
                 df['cast_list'].apply(lambda x: " ".join(x) * 2) + " " + \
                 df['director_list'].apply(lambda x: " ".join(x) * 3)
    
    df['tags'] = df['tags'].str.lower()
    return df

@st.cache_data(show_spinner=False)
def build_similarity_matrix(df):
    cv = TfidfVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return similarity

# --- CORE LOGIC ---
def recommend_movies(title, df, similarity, n=4):
    try:
        match = process.extractOne(title, df['title'])
        if match and match[1] > 70:
            idx = df[df['title'] == match[0]].index[0]
            distances = similarity[idx]
            movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:n+1]
            return [df.iloc[i[0]].to_dict() for i in movies_list]
    except Exception:
        pass
    return df.sort_values('popularity', ascending=False).head(n).to_dict('records')

def recommend_by_mood(mood, df, n=4):
    mood_map = {
        "happy": ["comedy", "animation", "family"],
        "sad": ["drama", "romance"],
        "excited": ["action", "thriller", "sciencefiction"],
        "scared": ["horror", "mystery", "thriller"],
        "inspired": ["history", "documentary"],
        "romantic": ["romance", "drama", "music"],
        "adventurous": ["adventure", "fantasy", "western"]
    }
    genres = mood_map.get(mood.lower(), [])
    if not genres:
        return df.sort_values('popularity', ascending=False).head(n).to_dict('records')
        
    df_copy = df.copy()
    df_copy['mood_score'] = df_copy['genres_list'].apply(lambda x: sum(1 for g in x if g in genres))
    res = df_copy[df_copy['mood_score'] > 0].sort_values(by=['mood_score', 'vote_average', 'popularity'], ascending=False)
    
    if len(res) == 0:
        return df.sort_values('popularity', ascending=False).head(n).to_dict('records')
    return [row.to_dict() for _, row in res.head(n).iterrows()]

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_poster(movie_id, title=None):
    if TMDB_API_KEY != "YOUR_API_KEY_HERE":
        try:
            url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
            data = requests.get(url, timeout=3).json()
            poster_path = data.get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
        except:
            pass
            
    # Fallback to free OMDB Public API if TMDB fails or key not set
    if title:
        try:
            clean_title = title.split('(')[0].strip()
            url = f"http://www.omdbapi.com/?apikey=trilogy&t={urllib.parse.quote(clean_title)}"
            data = requests.get(url, timeout=2).json()
            poster = data.get('Poster')
            if poster and poster != 'N/A':
                return poster
        except:
            pass

    colors = ["#e94560", "#00d4ff", "#ffd700", "#0d0221", "#4caf50"]
    c1 = colors[movie_id % len(colors) if isinstance(movie_id, int) else 0]
    c2 = colors[(movie_id + 1) % len(colors) if isinstance(movie_id, int) else 1]
    initials = "".join([w[0] for w in str(title).split()[:2]]).upper() if title else "M"
    return {"type": "gradient", "c1": c1, "c2": c2, "initials": initials}

# --- UI COMPONENTS ---
def render_movie_card(movie, index=0):
    movie_id = movie.get('id_x') if 'id_x' in movie else movie.get('id', 0)
    title = movie.get('title', 'Unknown')
    release = str(movie.get('release_date', ''))[:4]
    rating = round(movie.get('vote_average', 0), 1)
    overview = movie.get('overview', '')
    genres = movie.get('genres_list', [])[:3]
    
    poster = fetch_poster(movie_id, title)
    
    if isinstance(poster, dict):
        poster_html = f"""
<div style="width: 100%; height: 100%; background: linear-gradient(135deg, {poster['c1']}, {poster['c2']}); display: flex; align-items: center; justify-content: center;">
    <h1 style="color: white; font-size: 3rem; margin: 0; opacity: 0.6; font-family: 'Orbitron', sans-serif;">{poster['initials']}</h1>
</div>
"""
    else:
        poster_html = f'<img src="{poster}" class="poster-img" alt="{title}">'

    genres_html = "".join([f'<span class="genre-chip">{str(g).capitalize()}</span>' for g in genres])
    
    title_encoded = urllib.parse.quote(title)
    
    return f"""
<div class="movie-card" style="--card-index: {index};">
    <div class="poster-container">
        {poster_html.strip()}
    </div>
    <div class="card-content">
        <h3 class="movie-title" title="{title}">{title}</h3>
        <div class="movie-meta">
            <span class="year">{release}</span>
            <span class="rating">★ {rating}</span>
        </div>
        <div class="genre-chips">
            {genres_html}
        </div>
        <div class="movie-overview">
            {overview}
        </div>
        <a href="?movie={title_encoded}" target="_self" class="card-btn" style="display:block; text-decoration:none;">More Like This ▶</a>
    </div>
</div>
"""

def render_grid(recs):
    if not recs: return
    cols = st.columns(4)
    for i, movie in enumerate(recs):
        with cols[i % 4]:
            st.markdown(render_movie_card(movie, i), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

def apply_theme():
    inject_css()

def render_sidebar():
    with st.sidebar:
        st.markdown("<h1 class='orbitron' style='color: var(--aurora); text-align: center; text-shadow: 0 0 10px rgba(0,212,255,0.5);'>CineVerse AI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #888; font-size: 0.8rem; margin-top: -10px;'>Antigravity Edition</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.active_section = 'Home'
            st.session_state.selected_movie = None
            
        if st.button("🔥 Trending", use_container_width=True):
            st.session_state.active_section = 'Trending'
            
        if st.button("⭐ Top Rated", use_container_width=True):
            st.session_state.active_section = 'Top Rated'
            
        if st.button("🎭 By Mood", use_container_width=True):
            st.session_state.active_section = 'Mood'
            
        if st.button("🔍 Search", use_container_width=True):
            st.session_state.active_section = 'Search'
        
        st.markdown("<br><hr style='border-color: rgba(255,255,255,0.1);'><br>", unsafe_allow_html=True)
        
        st.markdown("<p style='font-family: Orbitron; font-size: 0.9rem;'>🌙 Toggle Universe Mode</p>", unsafe_allow_html=True)
        theme = st.radio("Theme", ["Dark Void", "Light Nebula"], label_visibility="collapsed")
        st.session_state.theme_mode = 'dark' if 'Dark' in theme else 'light'

# --- MAIN APP ---
def main():
    apply_theme()
    render_sidebar()
    
    try:
        with st.spinner("🌌 Calibrating the Cosmos... Downloading Dataset"):
            movies, credits = load_data()
            if 'processed_movies' not in st.session_state:
                st.session_state.processed_movies = preprocess_features(movies)
            df = st.session_state.processed_movies
            if 'similarity_matrix' not in st.session_state:
                st.session_state.similarity_matrix = build_similarity_matrix(df)
            similarity = st.session_state.similarity_matrix
    except Exception as e:
        st.error(f"Failed to load universe: {e}")
        st.markdown('<div class="black-hole-loader"></div>', unsafe_allow_html=True)
        return

    section = st.session_state.active_section

    if section == 'Home':
        st.markdown("""
        <div class="hero-banner reveal">
            <h1 class="orbitron" style="font-size: 3rem; margin: 0; color: var(--text);">Explore The Cinematic Universe</h1>
            <p style="font-size: 1.2rem; opacity: 0.8;">Physics-driven recommendations powered by Antigravity AI.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h2 class='orbitron reveal' style='margin-top: 20px;'>🚀 Trending Across The Universe</h2>", unsafe_allow_html=True)
        recs = [row.to_dict() for _, row in df.sort_values('popularity', ascending=False).head(8).iterrows()]
        render_grid(recs)

    elif section == 'Trending':
        st.markdown("<h2 class='orbitron reveal'>🔥 Trending Now</h2>", unsafe_allow_html=True)
        recs = [row.to_dict() for _, row in df.sort_values('popularity', ascending=False).head(20).iterrows()]
        render_grid(recs)

    elif section == 'Top Rated':
        st.markdown("<h2 class='orbitron reveal'>⭐ Top Rated of All Time</h2>", unsafe_allow_html=True)
        # Filter for movies with enough votes
        recs = [row.to_dict() for _, row in df[df['vote_count'] > 1000].sort_values('vote_average', ascending=False).head(20).iterrows()]
        render_grid(recs)

    elif section == 'Search':
        st.markdown("<h2 class='orbitron reveal'>🔍 Search the Galaxy</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        with col1:
            search_query = st.selectbox("Search (By Movie)", df['title'].dropna().unique(), index=None, placeholder="Type a movie title...")
        with col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🚀 Warp Speed (Recommend)", use_container_width=True) and search_query:
                st.session_state.selected_movie = search_query
                st.session_state.active_section = 'Recommended'
                st.rerun()

    elif section == 'Mood':
        st.markdown("<h2 class='orbitron reveal'>🎭 Mood Board</h2>", unsafe_allow_html=True)
        moods = [
            ("Happy 😄", "happy"), ("Sad 😢", "sad"), ("Excited 🤩", "excited"),
            ("Scared 😱", "scared"), ("Inspired 🌟", "inspired"), 
            ("Romantic ❤️", "romantic"), ("Adventurous 🌍", "adventurous")
        ]
        mood_cols = st.columns(len(moods))
        for i, (label, val) in enumerate(moods):
            with mood_cols[i]:
                if st.button(label, use_container_width=True):
                    st.session_state.current_mood = val
                    st.session_state.selected_movie = None
                    
        if st.session_state.current_mood:
            st.markdown(f"<p class='reveal' style='margin-top:20px; font-size:1.2rem'>Matching your <b>{st.session_state.current_mood}</b> mood...</p>", unsafe_allow_html=True)
            recs = recommend_by_mood(st.session_state.current_mood, df, n=12)
            render_grid(recs)

    elif section == 'Recommended':
        if st.session_state.selected_movie:
            st.markdown(f"<h2 class='orbitron reveal'>🤖 Because you liked {st.session_state.selected_movie}...</h2>", unsafe_allow_html=True)
            recs = recommend_movies(st.session_state.selected_movie, df, similarity, n=12)
            render_grid(recs)
        else:
            st.info("No movie selected. Please go to Search or Home to select a movie.")

if __name__ == "__main__":
    main()
