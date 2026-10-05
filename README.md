\# 🎬 CineMatch: Content-Based Movie Recommendation Engine



An end-to-end movie recommendation engine powered by natural language processing, metadata vectorization, and cosine angular similarity, served via an interactive Streamlit UI.



\---



\## 📌 Architecture \& Methodology



1\. \*\*Metadata Soup Generation:\*\* Aggregates categorical entities (`Director`, `Cast`, `Genres`, `Keywords`) and free-text plot summaries (`Overview`). High-signal fields such as the director and primary actors are weighted double to prioritize creator style and casting synergy.

2\. \*\*Vector Space Modeling:\*\* Employs `CountVectorizer` (with English stop-word filtering) to tokenize and map metadata soups into high-dimensional term-frequency vectors.

3\. \*\*Cosine Similarity Engine:\*\* Calculates pairwise dot products between normalized movie vectors:

&#x20;  $$\\text{Cosine Similarity}(A, B) = \\frac{A \\cdot B}{\\Vert{}A\\Vert{} \\Vert{}B\\Vert{}}$$

4\. \*\*Interactive UI Layer:\*\* Built using Streamlit, featuring title autocomplete, dynamic top-N recommendations, similarity match scores, and metadata inspection.



\---



\## 📁 Repository Structure



```text

movie-recommender-engine/

│

├── data/

│   ├── movies.csv             # Movie catalog with metadata and poster links

│   ├── movie\_list.pkl         # Pickled DataFrame for low-latency UI lookups

│   └── similarity.pkl         # Serialized pairwise cosine similarity matrix

├── src/

│   ├── generate\_movies.py     # Blockbuster catalog generator

│   └── recommend.py           # Preprocessing, vectorizer, and similarity calculations

├── app.py                     # Streamlit frontend dashboard

├── requirements.txt           # Environment dependencies

└── README.md                  # Project documentation

