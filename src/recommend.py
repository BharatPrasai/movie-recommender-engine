import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

def clean_text(text):
    """Lowercases text and collapses spaces between names (e.g. 'Tom Hardy' -> 'tomhardy')"""
    if isinstance(text, str):
        return " ".join([token.lower().replace(" ", "") for token in text.split()])
    return ""

def create_metadata_soup(df):
    """Combines weighted director, cast, genres, keywords, and overview into one token string."""
    df = df.copy()

    clean_director = df["director"].apply(clean_text)
    clean_cast = df["cast"].apply(clean_text)
    clean_genres = df["genres"].apply(lambda x: x.lower() if isinstance(x, str) else "")
    clean_keywords = df["keywords"].apply(lambda x: x.lower() if isinstance(x, str) else "")
    clean_overview = df["overview"].apply(lambda x: x.lower() if isinstance(x, str) else "")

    # Weight director and primary cast members double
    df["soup"] = (
        clean_director + " " + clean_director + " " +
        clean_cast + " " + clean_cast + " " +
        clean_genres + " " +
        clean_keywords + " " +
        clean_overview
    )
    return df

def build_similarity_matrix():
    data_path = "data/movies.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[!] '{data_path}' not found. Run generate_movies.py first.")

    df = pd.read_csv(data_path)
    df = create_metadata_soup(df)

    # Convert soup text into count vectors
    cv = CountVectorizer(stop_words="english", max_features=5000)
    count_matrix = cv.fit_transform(df["soup"])

    # Compute pairwise angular distance
    similarity = cosine_similarity(count_matrix, count_matrix)

    # Save artifacts for Streamlit UI
    with open("data/movie_list.pkl", "wb") as f:
        pickle.dump(df, f)

    with open("data/similarity.pkl", "wb") as f:
        pickle.dump(similarity, f)

    print(f"[+] Computed similarity matrix shape: {similarity.shape}")
    print("[+] Successfully exported 'data/movie_list.pkl' and 'data/similarity.pkl'")
    return df, similarity

def get_recommendations(title, top_n=5):
    """Query top-N similar movies given a title."""
    with open("data/movie_list.pkl", "rb") as f:
        df = pickle.load(f)

    with open("data/similarity.pkl", "rb") as f:
        similarity = pickle.load(f)

    matches = df[df["title"].str.lower() == title.lower()]
    if matches.empty:
        print(f"[!] Movie '{title}' not found in catalog.")
        return []

    idx = matches.index[0]
    sim_scores = list(enumerate(similarity[idx]))

    # Sort descending by similarity score, skipping the queried movie itself
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

    recommendations = []
    for item in sim_scores:
        m_idx = item[0]
        score = item[1]
        recommendations.append({
            "title": df.iloc[m_idx]["title"],
            "genres": df.iloc[m_idx]["genres"],
            "director": df.iloc[m_idx]["director"],
            "vote_average": df.iloc[m_idx]["vote_average"],
            "similarity_score": round(score * 100, 2)
        })

    return recommendations

if __name__ == "__main__":
    build_similarity_matrix()
    test_title = "Inception"
    print(f"\n--- Top 5 Recommendations for '{test_title}' ---")
    results = get_recommendations(test_title, top_n=5)
    for r in results:
        print(f"• {r['title']} ({r['genres']}) | Dir: {r['director']} | Match: {r['similarity_score']}%")
