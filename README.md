# CodeOrbit Content-Based Recommendation System

## Project Overview & Purpose
A content-based movie recommendation engine (`recommender.py`) built with **scikit-learn**, **pandas**, and **NumPy** over a curated 20-title catalog (`movies.csv`). Supports both **item-to-item** similarity lookup and **composite user-profile** recommendations without requiring collaborative rating matrices or external user logs.

---

## How It Works (TF-IDF & Cosine Similarity)
1. **Metadata Feature Construction:** Loads `movies.csv` (`title`, `genres`, `description`), fills any missing text gracefully, and concatenates `genres + " " + description` into a unified document string per item.
2. **TF-IDF Vectorization:** Fits `TfidfVectorizer(stop_words="english")` across the catalog, transforming text descriptors into a sparse term-frequency inverse-document-frequency matrix that downweights common English words and highlights distinctive genre/plot terms.
3. **Cosine Similarity Matrix:** Computes normalized angular alignment between item vectors:
   $$\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$
   Scores fall in $[0.0, 1.0]$ and remain invariant to description length.
4. **Item-to-Item Search (`recommend_similar`):** Performs case-insensitive title matching, sorts the item's similarity row descending, excludes the query item itself, and returns the Top $N$ neighbors.
5. **User Profile Centroid (`recommend_for_user`):** Averages the TF-IDF vectors of all recognized liked titles (`np.asarray(liked_vectors.mean(axis=0))`), scores every catalog item against this centroid vector, filters out already-liked titles, and returns the Top $N$ recommendations.

---

## Setup & How to Run

1. Navigate into the project directory:
   ```bash
   cd CodeOrbit_RecommendationSystem
   ```
2. Activate the virtual environment (or install from `requirements.txt`):
   - **Windows PowerShell:**
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - **Git Bash / Linux / macOS:**
     ```bash
     source .venv/Scripts/activate
     ```
   *(Fresh setup: `pip install -r requirements.txt`)*
3. Run the recommendation demo and verification matrix:
   ```bash
   python recommender.py
   python test_recommender.py
   ```

---

## Example Output

```text
============================================================
  CONTENT-BASED RECOMMENDATION ENGINE INITIALIZED           
  Catalog Size: 20 items
============================================================

Top 5 Recommendations for fans of 'Inception':
  1. The Matrix                (Similarity Score: 0.0909)
  2. The Dark Knight           (Similarity Score: 0.0729)
  3. Arrival                   (Similarity Score: 0.0582)
  4. Interstellar              (Similarity Score: 0.0545)
  5. WALL-E                    (Similarity Score: 0.0514)

Top 5 Recommendations for User with profile liked: ['The Matrix', 'Blade Runner 2049']
  1. Arrival                   (Similarity Score: 0.1523)
  2. WALL-E                    (Similarity Score: 0.0959)
  3. Inception                 (Similarity Score: 0.0928)
  4. Interstellar              (Similarity Score: 0.0749)
  5. Se7en                     (Similarity Score: 0.0585)
------------------------------------------------------------
```
