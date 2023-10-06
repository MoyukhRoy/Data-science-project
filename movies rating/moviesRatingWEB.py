import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

# Load the datasets
movies = pd.read_csv('C:\\Users\\user\\PycharmProjects\\pythonProject\\DIY Dataset\\movies.csv')
ratings = pd.read_csv('C:\\Users\\user\\PycharmProjects\\pythonProject\\DIY Dataset\\ratings.csv')

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

def content_recommender(movie_title, N):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies['genres'])

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    idx = movies[movies['title'].str.lower() == movie_title.lower()].index[0]
    similar_movies = list(enumerate(cosine_sim[idx]))
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:N+1]

    recommendations = [(movies.iloc[i[0]]['title'], i[1]) for i in similar_movies]
    return recommendations

def collaborative_recommender(user_id, N, k):
    user_ratings = ratings[ratings['userId'] == user_id]
    user_movie_ratings = user_ratings.merge(movies, on='movieId')

    user_movie_matrix = user_movie_ratings.pivot_table(index='userId', columns='title', values='rating')

    similar_users = cosine_similarity(user_movie_matrix, user_movie_matrix)[user_id - 1]
    similar_users_idx = sorted(range(len(similar_users)), key=lambda i: similar_users[i], reverse=True)[1:k+1]

    recommendations = user_movie_matrix.iloc[similar_users_idx].mean().sort_values(ascending=False).head(N)
    return recommendations

# Streamlit app
st.title('Movie Recommender System')

# Sidebar inputs
genre = st.text_input('Enter Genre:')
threshold = st.slider('Minimum Threshold:', min_value=1, max_value=1000, value=100)
num_recommendations = st.slider('Number of Recommendations:', min_value=1, max_value=20, value=5)

# Recommendation type selection
recommendation_type = st.radio('Select Recommendation Type:', ['Popularity', 'Content-Based', 'Collaborative'])

if st.button('Recommend'):
    if recommendation_type == 'Popularity':
        recommendations = popularity_recommender(genre, threshold, num_recommendations)
    elif recommendation_type == 'Content-Based':
        movie_title = st.text_input('Enter Movie Title:')
        recommendations = content_recommender(movie_title, num_recommendations)
    elif recommendation_type == 'Collaborative':
        user_id = st.number_input('Enter User ID:', min_value=1, max_value=unique_users)
        recommendations = collaborative_recommender(user_id, num_recommendations, 100)

    st.subheader('Recommendations:')
    st.write(recommendations)
