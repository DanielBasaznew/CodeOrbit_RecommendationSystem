from recommender import ContentRecommender

recommender = ContentRecommender("movies.csv")

print("=" * 65)
print("              HOUR 4 VERIFICATION TEST MATRIX                 ")
print("=" * 65)

# Test 1: Sci-Fi Seed Item
print("\n[Test 1] Sci-Fi Seed: 'Interstellar' (Top 3)")
for rank, (title, score) in enumerate(recommender.recommend_similar("Interstellar", n=3), 1):
    print(f"  {rank}. {title:<25} ({score:.4f})")

# Test 2: Romance Seed Item
print("\n[Test 2] Romance Seed: 'The Notebook' (Top 3)")
for rank, (title, score) in enumerate(recommender.recommend_similar("The Notebook", n=3), 1):
    print(f"  {rank}. {title:<25} ({score:.4f})")

# Test 3: Crime Seed Item
print("\n[Test 3] Crime Seed: 'The Godfather' (Top 3)")
for rank, (title, score) in enumerate(recommender.recommend_similar("The Godfather", n=3), 1):
    print(f"  {rank}. {title:<25} ({score:.4f})")

# Test 4: Case-Insensitivity Check
print("\n[Test 4] Case-Insensitive Lookup: 'inception' (all lower)")
recs_lower = recommender.recommend_similar("inception", n=2)
for rank, (title, score) in enumerate(recs_lower, 1):
    print(f"  {rank}. {title:<25} ({score:.4f})")

# Test 5: Unknown Title Graceful Handling
print("\n[Test 5] Nonexistent Title Handling: 'Avatar 3'")
recommender.recommend_similar("Avatar 3", n=3)

# Test 6: Oversized N (N=50 on a 20-item catalog)
print("\n[Test 6] Oversized n=50 on 'WALL-E':")
recs_oversized = recommender.recommend_similar("WALL-E", n=50)
print(f"  Returned {len(recs_oversized)} recommendations (Catalog max: {len(recommender.df) - 1}).")

# Test 7: Multi-Genre User Profile (Sci-Fi + Romance hybrid)
print("\n[Test 7] Mixed-Genre User Profile: ['Interstellar', 'Titanic']")
user_recs = recommender.recommend_for_user(["Interstellar", "Titanic"], n=4)
for rank, (title, score) in enumerate(user_recs, 1):
    print(f"  {rank}. {title:<25} ({score:.4f})")

print("=" * 65)