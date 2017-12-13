import os

import numpy
import pandas
from tqdm import tqdm
DATA_PATH = '/Users/sai/dev/datasets/movielens-20m/ml-20m/'
NUM_USERS, NUM_MOVIES = 138493, 131262
BATCH_SIZE = 25000


def get_ratings_data():
    ratings_file = os.path.join(DATA_PATH, 'ratings.csv')
    data = pandas.read_csv(ratings_file, sep=',', usecols=(0, 1, 2))

    print("user id min/max: ", data['userId'].min(), data['userId'].max())
    # assert numpy.unique(data['userId']).shape[0] == NUM_USERS
    print "Number of unique users: {}".format(NUM_USERS)
    print("movie id min/max: ", data['movieId'].min(), data['movieId'].max())
    # assert numpy.unique(data['movieId']).shape[0] == NUM_MOVIES
    print "Number of unique movies: {}".format(NUM_MOVIES)

    data = data.sample(frac=1).reset_index(drop=True)  # Shuffle the data in place row-wise

    # Use the first 19M samples to train the model
    train_users = data['userId'].values - 1  # Offset by 1 so that the IDs start at 0
    train_movies = data['movieId'].values - 1  # Offset by 1 so that the IDs start at 0
    train_ratings = data['rating'].values

    return train_users, train_movies, train_ratings


def get_genres_data(train_movies):
    genres_file = os.path.join(DATA_PATH, 'movies.csv')
    data = pandas.read_csv(genres_file, sep=',', usecols=(0, 1, 2))
    movie_id_arr = data['movieId'].values - 1
    movie_id_list = movie_id_arr.tolist()
    movie_id_to_idx = {}
    for movie_id in tqdm(movie_id_list):
        movie_id_to_idx[movie_id] = movie_id_list.index(movie_id)
    GENRES = [
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "IMAX",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
        "(no genres listed)"
    ]
    num_genres = len(GENRES)

    def multi_hot_udf(g):
        arr = numpy.zeros(num_genres)
        l = g.split('|')
        for i in l:
            arr[GENRES.index(i)] = 1
        return arr

    d = data['genres'].apply(multi_hot_udf)
    arr_genres = numpy.array(d.tolist())
    arr_genres_dataset = []
    for movie_id in tqdm(train_movies):
        arr_genres_dataset.append(arr_genres[movie_id_to_idx[movie_id]])
    return numpy.array(arr_genres_dataset)


def transform_ratings_into_classes(ratings):
    num_rows = ratings.shape[0]
    t = 2 * ratings - 1
    t = t.astype('int32')
    b = numpy.zeros((num_rows, 10))
    b[numpy.arange(num_rows), t] = 1
    return b

if __name__ == "__main__":
    train_users, train_movies, train_ratings = get_ratings_data()
    train_genres = get_genres_data(train_movies)
    from IPython import embed
    embed()
