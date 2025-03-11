import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def movie_recommendations(title, movies_df, num_recommendations=5):
    # Fill missing descriptions
    movies_df['description'] = movies_df['description'].fillna('')

    # Convert descriptions to TF-IDF vectors
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['description'])

    # Compute similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Get movie index
    indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()
    if title not in indices:
        return f"Movie '{title}' not found in database."

    idx = indices[title]

    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top recommendations
    sim_scores = sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]

    return movies_df['title'].iloc[movie_indices]

# Sample dataset
movies_data = {
    'title': ['Dangal', 'Zindagi Na Milegi Dobara', '3 Idiots', 'Gully Boy', 'Chak De! India'],
    'description': [
        'A father trains his daughters to become world-class wrestlers.',
        'Three friends rediscover themselves on a road trip.',
        'Two friends search for their missing friend while recalling college days.',
        'A street rapper rises to fame despite struggles.',
        'A coach leads an underdog women’s hockey team to victory.'
    ]
}

movies_df = pd.DataFrame(movies_data)

# Get recommendations
movie_name = 'Dangal'
recommendations = movie_recommendations(movie_name, movies_df)
print(f"Movies recommended for '{movie_name}':\n", recommendations)
