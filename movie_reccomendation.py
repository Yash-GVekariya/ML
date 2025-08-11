'''
5. Movie Recommendation System (Content-Based)

Description: Recommend movies to users based on movie descriptions or genres using cosine 
similarity. 

Steps: 

• Load a movie dataset with titles and descriptions (e.g., TMDB dataset). 

• Preprocess text (cleaning, vectorization using TF-IDF). 

• Compute cosine similarity matrix. 

• Define a function to recommend similar movies based on input title. 

• Display top 5 recommendations. 
'''

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas

    # loading dataset (TMDB dataset)
df = pandas.read_csv('Data Analytics\\Practise Problems\\tmdb_5000_credits.csv\\tmdb_5000_credits.csv')
    # print(df.isnull().sum()) #To check is their any null value int our dataset

    # Computing the similarity by TF-IDF Vectorizer
df['title'] = df['title'].fillna('')
tf_vector = TfidfVectorizer(stop_words='english')

tf_matirx = tf_vector.fit_transform(df['title'])
    # print(tf_matirx.shape)

cos_sim = cosine_similarity(tf_matirx, tf_matirx)
    # print(cos_sim.shape)

# Track the index of the movie  
indices = pandas.Series(df.index, index=df['title']).drop_duplicates()

def recommend(title, cosine_sim_matrix, num_of_recommendations=5,):
    # This is a function which will recommend the movies

    # Get the index of the movie that matches the title
    try:
        idx = indices[title]
    except KeyError:
        return "Movie not found. Please check the spelling."

    # Get the similarity scores for all movies with that movie
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top N most similar movies (excluding itself)
    sim_scores = sim_scores[1:num_of_recommendations+1]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top N recommended movies
    return df['title'].iloc[movie_indices]

# Testing the model
recommendation = recommend('Avatar', cos_sim)
print(recommendation)
