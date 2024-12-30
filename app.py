import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Fill missing genres
movies['genres'] = movies['genres'].fillna('')

# Preprocess genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create user-movie matrix
user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')

# Recommendation Functions
def recommend_content_based(title, movies_df, similarity_matrix):
    # Get the index of the movie
    try:
        idx = movies_df[movies_df['title'].str.contains(title, case=False, na=False)].index[0]
    except IndexError:
        raise IndexError("Movie not found in the dataset.")
    
    # Get similarity scores
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    
    # Get the recommended movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top 10 most similar movies
    return movies_df.iloc[movie_indices][['title', 'genres']]

def recommend_collaborative(user_id, user_movie_matrix):
    user_ratings = user_movie_matrix.loc[user_id].dropna()
    user_sim = user_movie_matrix.corrwith(user_ratings).dropna().sort_values(ascending=False)
    
    recommendations = {}
    for other_user, sim_score in user_sim.items():
        if other_user == user_id:
            continue
        other_ratings = user_movie_matrix.loc[other_user].dropna()
        for movie, rating in other_ratings.items():
            if movie not in user_ratings.index:
                recommendations[movie] = recommendations.get(movie, 0) + rating * sim_score
    
    sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:10]
    return [movies[movies['movieId'] == movie]['title'].values[0] for movie, _ in sorted_recs]

# Streamlit App
st.title("Movie Recommendation System")

# Input for content-based recommendations
movie_title = st.text_input("Enter a movie title:")
if movie_title:
    try:
        content_recommendations = recommend_content_based(movie_title, movies, cosine_sim)
        st.write("Content-Based Recommendations:")
        st.table(content_recommendations)
    except IndexError:
        st.error("Movie not found. Try another title.")

# Input for collaborative filtering
user_id = st.number_input("Enter a User ID:", min_value=1, step=1)
if user_id:
    try:
        collaborative_recommendations = recommend_collaborative(user_id, user_movie_matrix)
        st.write("Collaborative Recommendations:")
        st.write(collaborative_recommendations)
    except KeyError:
        st.error("User ID not found. Try a valid User ID.")
