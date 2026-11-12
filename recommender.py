import os
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:
    def __init__(self, data_path: str = "movies.csv"):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Catalog file '{data_path}' was not found!")

        self.df = pd.read_csv(data_path)
        # Handle possible missing text values gracefully
        self.df["genres"] = self.df["genres"].fillna("")
        self.df["description"] = self.df["description"].fillna("")

        # 1. Combine metadata into single feature string
        self.df["features"] = self.df["genres"] + " " + self.df["description"]

        # 2. Extract TF-IDF sparse matrix
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["features"])

        # 3. Compute pairwise cosine similarity matrix
        self.sim_matrix = cosine_similarity(self.tfidf_matrix)

        # Mapping dictionary for case-insensitive lookup: "inception" -> row_index
        self.title_to_idx = {
            title.strip().lower(): idx for idx, title in enumerate(self.df["title"])
        }

    def recommend_similar(self, title: str, n: int = 5) -> List[Tuple[str, float]]:
        """
        Recommends top N items similar to a given title based on cosine similarity.
        Gracefully handles unknown titles, case differences, and oversized n.
        """
        lookup = title.strip().lower()
        if lookup not in self.title_to_idx:
            print(f"[Notice] Movie '{title}' was not found in the catalog.")
            return []

        target_idx = self.title_to_idx[lookup]

        # Retrieve similarity scores for this item across all catalog entries
        scores = list(enumerate(self.sim_matrix[target_idx]))

        # Sort descending by similarity score
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        # Filter out the item itself and collect top N
        recommendations = []
        for idx, score in scores:
            if idx == target_idx:
                continue
            item_title = self.df.iloc[idx]["title"]
            recommendations.append((item_title, float(score)))
            if len(recommendations) == n:
                break

        return recommendations

    def recommend_for_user(
        self, liked_titles: List[str], n: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Builds a composite user profile by averaging TF-IDF vectors of liked movies,
        then recommends the top N unconsumed items closest to the user's centroid.
        """
        valid_indices = []
        for t in liked_titles:
            clean_t = t.strip().lower()
            if clean_t in self.title_to_idx:
                valid_indices.append(self.title_to_idx[clean_t])
            else:
                print(f"[Warning] Liked title '{t}' not recognized; skipping.")

        if not valid_indices:
            print("[Notice] None of the liked titles were found. Cannot build profile.")
            return []

        # Average the TF-IDF vectors of liked items
        liked_vectors = self.tfidf_matrix[valid_indices]
        profile_mean = liked_vectors.mean(axis=0)

        # Convert matrix centroid to numpy array to prevent dimension/type errors
        profile_arr = np.asarray(profile_mean)

        # Compute cosine similarity between the user profile and all items
        user_scores = cosine_similarity(profile_arr, self.tfidf_matrix)[0]

        # Pair scores with movie indices
        indexed_scores = list(enumerate(user_scores))
        indexed_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)

        # Exclude already-liked items
        liked_indices_set = set(valid_indices)
        recommendations = []
        for idx, score in indexed_scores:
            if idx in liked_indices_set:
                continue
            rec_title = self.df.iloc[idx]["title"]
            recommendations.append((rec_title, float(score)))
            if len(recommendations) == n:
                break

        return recommendations


# ==============================================================================
# Sample Verification & Demonstration
# ==============================================================================
if __name__ == "__main__":
    recommender = ContentRecommender("movies.csv")
    print("=" * 60)
    print("  CONTENT-BASED RECOMMENDATION ENGINE INITIALIZED           ")
    print(f"  Catalog Size: {len(recommender.df)} items")
    print("=" * 60)

    # 1. Single Item Recommendation Demo (Inception)
    target_movie = "Inception"
    print(f"\nTop 5 Recommendations for fans of '{target_movie}':")
    recs = recommender.recommend_similar(target_movie, n=5)
    for rank, (title, score) in enumerate(recs, start=1):
        print(f"  {rank}. {title:<25} (Similarity Score: {score:.4f})")

    # 2. Composite User Profile Demo (User who likes multiple Sci-Fi films)
    liked_list = ["The Matrix", "Blade Runner 2049"]
    print(f"\nTop 5 Recommendations for User with profile liked: {liked_list}")
    user_recs = recommender.recommend_for_user(liked_list, n=5)
    for rank, (title, score) in enumerate(user_recs, start=1):
        print(f"  {rank}. {title:<25} (Similarity Score: {score:.4f})")
    print("-" * 60)