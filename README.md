# CodeOrbit Content-Based Recommendation System (Day 7)

A content-based recommendation engine built in Python using TF-IDF feature vectorization and Cosine Similarity over metadata descriptors.

## How It Works
1. **Feature Extraction:** Concatenates item metadata (`genres` and `description`) into a unified text feature string per title.
2. **TF-IDF Vectorization:** Transforms raw text features into a sparse term-frequency inverse-document-frequency matrix using `TfidfVectorizer(stop_words='english')`, downweighting ubiquitous vocabulary.
3. **Cosine Similarity Metric:** Computes angular alignment between vectors via $\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\Vert{}\mathbf{A}\Vert{} \Vert{}\mathbf{B}\Vert{}}$. Scores range between $0.0$ (disjoint vocabulary) and $1.0$ (identical vocabulary orientation), independent of text length.
4. **Item-to-Item Recommendations:** Queries the pairwise similarity matrix for a target title, sorts matches descending, excludes the query item itself, and extracts the Top $N$ records.
5. **Composite User Profile:** Given a list of consumed titles, the engine calculates the centroid vector (`np.asarray(matrix[idxs].mean(axis=0))`), scores all catalog candidates against the user's centroid, excludes already-liked items, and returns top recommendations.

---

## How to Run It

1. Activate the dedicated virtual environment:
   ```bash
   source .venv/Scripts/activate  # On Windows Git Bash


