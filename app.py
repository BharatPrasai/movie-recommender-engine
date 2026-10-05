import streamlit as st
import pickle
import pandas as pd

# Page setup
st.set_page_config(
    page_title="Movie Matchmaker",
    page_icon="🎬",
    layout="wide"
)

# Load serialized data
@st.cache_data
def load_data():
    with open("data/movie_list.pkl", "rb") as f:
        movies = pickle.load(f)
    with open("data/similarity.pkl", "rb") as f:
        similarity = pickle.load(f)
    return movies, similarity

movies, similarity = load_data()

# App Header
st.title("🎬 Intelligent Movie Recommendation Engine")
st.caption("Content-based similarity scoring using metadata soups (Cast, Director, Genres, Plot, and Keywords).")

# Sidebar Controls
st.sidebar.header("Recommendation Settings")
top_n = st.sidebar.slider("Number of recommendations:", min_value=3, max_value=8, value=5)

# Selection Dropdown
movie_list = sorted(movies["title"].tolist())
selected_movie = st.selectbox(
    "Type or select a movie you like:",
    movie_list,
    index=movie_list.index("Inception") if "Inception" in movie_list else 0
)

def recommend(movie_title, n):
    idx = movies[movies["title"] == movie_title].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
    
    recommendations = []
    # Skip index 0 because it's the movie itself
    for i in distances[1:n + 1]:
        row = movies.iloc[i[0]]
        recommendations.append({
            "title": row["title"],
            "genres": row["genres"],
            "director": row["director"],
            "cast": row["cast"],
            "vote_average": row["vote_average"],
            "overview": row["overview"],
            "score": round(i[1] * 100, 1)
        })
    return recommendations

# Selected Movie Profile Card
selected_row = movies[movies["title"] == selected_movie].iloc[0]
with st.expander("🔍 Selected Movie Details", expanded=True):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"**Director:** {selected_row['director']}")
        st.markdown(f"**Genres:** `{selected_row['genres']}`")
        st.markdown(f"**Rating:** ⭐ **{selected_row['vote_average']} / 10**")
    with col2:
        st.markdown(f"**Key Cast:** {selected_row['cast']}")
        st.markdown(f"**Overview:** {selected_row['overview']}")

st.divider()

# Trigger Recommendations
if st.button("✨ Show Similar Recommendations", type="primary"):
    recs = recommend(selected_movie, top_n)
    
    st.subheader(f"Top {top_n} Movies Similar to '{selected_movie}'")
    
    # Display results as columns / cards
    cols = st.columns(top_n)
    for idx, movie in enumerate(recs):
        with cols[idx]:
            st.markdown(f"### {idx + 1}. {movie['title']}")
            st.metric(label="Match Confidence", value=f"{movie['score']}%")
            st.markdown(f"⭐ **{movie['vote_average']}/10**")
            st.caption(f"**Director:** {movie['director']}")
            st.caption(f"**Genre:** {movie['genres']}")
            with st.popover("Plot Overview"):
                st.write(movie["overview"])
                st.write(f"**Cast:** {movie['cast']}")
