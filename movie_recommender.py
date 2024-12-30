import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Fill missing values in the 'genres' column
movies['genres'] = movies['genres'].fillna('')

# Create TF-IDF Matrix for genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Calculate the cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_content_based(title, movies_df, similarity_matrix):
    # Get the index of the movie
    idx = movies_df[movies_df['title'] == title].index[0]
    
    # Get similarity scores
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    
    # Get the recommended movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top 10 most similar movies
    return movies_df.iloc[movie_indices][['title', 'genres']]

# Example usage
recommendations = recommend_content_based('Toy Story (1995)', movies, cosine_sim)
print(recommendations)
