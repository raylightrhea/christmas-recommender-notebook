# app.py

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("christmas_tracks.csv")

# Build a TF-IDF matrix on titles
tfidf = TfidfVectorizer(stop_words="english")
tfidf_mat = tfidf.fit_transform(df["title"].fillna(""))

# Compute cosine similarity once
sim = cosine_similarity(tfidf_mat, tfidf_mat)

titles = df["title"].tolist()

# UI 
st.title("🎄 Christmas Carol Recommender (Title-Only)")

anchor = st.selectbox("Choose an anchor carol:", titles, index=titles.index("Jingle Bells"))
top_n = st.slider("Number of suggestions:", 1, 10, 5)

# Compute recommendations in real time
idx = titles.index(anchor)
scores = list(enumerate(sim[idx]))
scores = sorted(scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]

st.subheader(f"Top {top_n} matches for “{anchor}”")
for i, score in scores:
    row = df.iloc[i]
    st.write(f"**{row.title}** by *{row.artist}*  —  similarity {score:.2f}")

    # Embed full Spotify player for each track
    track_id = row['id']
    # Use Components API to embed the Spotify Player
    embed_url = f"https://open.spotify.com/embed/track/{track_id}"
    components.iframe(
        src=embed_url,
        width=300,
        height=80,
        scrolling=False
    )
    
