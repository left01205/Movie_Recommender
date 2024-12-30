# Movie Recommendation System

This project implements a Movie Recommendation System using both Content-Based Filtering and Collaborative Filtering techniques. The system allows users to find movies similar to a given title or receive personalized recommendations based on user ratings.

---

## Features

1. **Content-Based Recommendations:**

   - Uses genres and other movie metadata to recommend similar movies.
   - Powered by TF-IDF vectorization and cosine similarity.

2. **Collaborative Filtering:**

   - Leverages user-movie rating data to recommend movies based on user similarity.
   - Handles cases where users have overlapping preferences.

3. **Streamlit Web Interface:**

   - An interactive UI for users to enter movie titles or user IDs.
   - Displays recommendations in a tabular format.

---

## Prerequisites

1. **Python 3.8+**
2. Libraries:
   - pandas
   - scikit-learn
   - streamlit
   - numpy
   - fuzzywuzzy (optional, for title matching)
3. Datasets:
   - `movies.csv`: Contains movie information (e.g., title, genres).
   - `ratings.csv`: Contains user ratings for movies.

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repository/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. Install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Place the datasets (`movies.csv` and `ratings.csv`) in the project directory.

---

## Usage

### Run the Streamlit App:

```bash
streamlit run app.py
```

- Open the app in your browser at [http://localhost:8501](http://localhost:8501).
- **Content-Based Filtering:** Enter a movie title to receive similar movies.
- **Collaborative Filtering:** Enter a User ID to get personalized recommendations.

### Script Usage (CLI):

For testing the recommendation engine without the Streamlit interface, run:

```bash
python movie_recommender.py
```

Edit the example movie title in the script to test content-based recommendations.

---

## File Structure

```
movie-recommendation-system/
|
|-- app.py               # Main Streamlit app
|-- movie_recommender.py # Standalone recommendation script
|-- movies.csv           # Movie dataset
|-- ratings.csv          # Ratings dataset
|-- README.md            # Documentation
|-- requirements.txt     # Dependencies
```

---

## Optimization Suggestions

1. **Caching:**

   - Cache the TF-IDF matrix and cosine similarity computations for faster performance.

2. **Feature Enrichment:**

   - Incorporate additional metadata (e.g., cast, director) to improve recommendations.

3. **Matrix Factorization:**

   - Use SVD or ALS for collaborative filtering to handle large datasets efficiently.

4. **UI Enhancements:**

   - Add auto-complete for movie titles and display movie posters.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

- Datasets from the [MovieLens](https://grouplens.org/datasets/movielens/) project.
- Built using [Streamlit](https://streamlit.io/) and [scikit-learn](https://scikit-learn.org/).

---
