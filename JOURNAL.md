# Engineering Journal — Day 7: Recommendation Systems (Task 4)

## 1. What I Studied
- **Content-Based Filtering:** Recommending items using metadata attributes (genres, synopsis) without requiring interaction logs or other users' ratings.
- **TF-IDF Vector Space:** Converting concatenated text features into weighted sparse representations.
- **Cosine Metric Mechanics:** Calculating vector similarity via normalized dot products, making comparison invariant to description lengths.
- **Profile Centroids:** Aggregating multiple liked item vectors into an average centroid representation to model user taste.

## 2. Core Notes
1. **Vector Representation:** Items are encoded as vectors in high-dimensional word space; terms that are unique across the corpus carry higher weights.
2. **Cosine Invariance:** Unlike Euclidean distance, Cosine Similarity evaluates directional angle rather than magnitude, ensuring longer descriptions do not artificially lower similarity.
3. **Self-Exclusion:** Item-to-item recommendation loops must filter out the target item index ($idx == target\_idx$), which always has a similarity score of $1.00$.
4. **Centroid Alignment:** Averaging multiple sparse rows via `.mean(axis=0)` yields a matrix object that must be wrapped with `np.asarray()` to avoid matrix dimension errors during cosine computation.
5. **Consumption Filtering:** When computing user recommendations, all previously consumed items must be excluded from candidate consideration.

## 3. What I Built
- Set up `CodeOrbit_RecommendationSystem` with isolated `.venv` and frozen dependencies.
- Created `movies.csv` containing 20 curated titles across four thematic clusters (Sci-Fi, Animation, Romance, Crime).
- Implemented `ContentRecommender` in `recommender.py`:
  - Feature concatenation and TF-IDF matrix generation.
  - Pairwise cosine matrix computation.
  - Case-insensitive item similarity search (`recommend_similar`).
  - Centroid user profiling (`recommend_for_user`).
- Built automated test harness `test_recommender.py`.

## 4. Tests Run & Empirical Findings
- **Single-Genre Retrieval:** Sci-Fi seed *Interstellar* returned *WALL-E* (0.1694) and *Arrival* (0.0806). Romance seed *The Notebook* returned *Before Sunrise* (0.2152) and *Pride and Prejudice* (0.1822).
- **Hybrid Preferences:** Averaging *Interstellar* (Sci-Fi) and *Titanic* (Romance) successfully yielded top items spanning both spaces (*WALL-E*, *The Notebook*, *Arrival*).
- **Edge Conditions:** Handled oversized $N=50$ by returning all 19 other catalog items, and caught missing titles (`"Avatar 3"`) without halting runtime.

## 5. End-to-End Explanation (5 Lines)
1. `movies.csv` combines genre tags and descriptions into a unified `features` string.
2. `TfidfVectorizer` computes a numerical feature matrix across catalog vocabulary.
3. `cosine_similarity` evaluates pairwise dot products across normalized vectors.
4. `recommend_similar` retrieves an item's row, sorts descending, and drops self-matches.
5. `recommend_for_user` computes the mean vector across liked titles, scores all catalog items, removes consumed items, and returns the top $N$.

## 6. What I Don't Understand Yet / Next Steps
- Incorporating categorical weighting (e.g., placing double weight on matching genres versus plot synopsis).
- Hybrid recommender architectures that combine content-based TF-IDF with collaborative matrix factorization (SVD).

## 7. Upcoming Roadmap
- Complete technical roadmap evaluation and transition to deployment and API serving.