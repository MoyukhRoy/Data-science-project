

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
import ipywidgets as widgets
from IPython.display import display



# Load the datasets
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
print(movies)
print(ratings)

unique_users = ratings['userId'].nunique()
unique_movies = ratings['movieId'].nunique()

# Average rating and total movies at genre level
avg_rating_genre = ratings.merge(movies, on='movieId').groupby('genres')['rating'].mean()
total_movies_genre = movies['genres'].value_counts()

# Unique genres considered
unique_genres = movies['genres'].str.split('|', expand=True).stack().unique()

def popularity_recommender(genre, threshold, N):
    genre_movies = movies[movies['genres'].str.contains(genre, case=False)]
    popular_movies = genre_movies.merge(ratings, on='movieId').groupby('title').agg(
        {'rating': 'mean', 'userId': 'count'}
    )
    popular_movies = popular_movies[popular_movies['userId'] >= threshold].sort_values(
        by=['rating', 'userId'], ascending=[False, False]
    )
    return popular_movies.head(N)

# Example usage
pop_recommendations = popularity_recommender('Comedy', 100, 5)
print(pop_recommendations)

def content_recommender(movie_title, N):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies['genres'])

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    idx = movies[movies['title'].str.lower() == movie_title.lower()].index[0]
    similar_movies = list(enumerate(cosine_sim[idx]))
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:N+1]

    recommendations = [(movies.iloc[i[0]]['title'], i[1]) for i in similar_movies]
    return recommendations

# Example usage
content_recommendations = content_recommender('Toy Story (1995)', 5)
print(content_recommendations)

def collaborative_recommender(user_id, N, k):
    user_ratings = ratings[ratings['userId'] == user_id]
    user_movie_ratings = user_ratings.merge(movies, on='movieId')

    user_movie_matrix = user_movie_ratings.pivot_table(index='userId', columns='title', values='rating')

    similar_users = cosine_similarity(user_movie_matrix, user_movie_matrix)[user_id - 1]
    similar_users_idx = sorted(range(len(similar_users)), key=lambda i: similar_users[i], reverse=True)[1:k+1]

    recommendations = user_movie_matrix.iloc[similar_users_idx].mean().sort_values(ascending=False).head(N)
    return recommendations

# Example usage
collab_recommendations = collaborative_recommender(1, 5, 100)
print(collab_recommendations)

genre_input = widgets.Text(description="Genre:")
threshold_input = widgets.IntSlider(description="Minimum Threshold:", min=1, max=1000, value=100)
num_input = widgets.IntSlider(description="Num Recommendations:", min=1, max=20, value=5)

def recommend(genre, threshold, num):
    recommendations = popularity_recommender(genre, threshold, num)
    display(recommendations)

recommend_button = widgets.Button(description="Recommend")
recommend_button.on_click(lambda _: recommend(genre_input.value, threshold_input.value, num_input.value))

widgets.VBox([genre_input, threshold_input, num_input, recommend_button])
