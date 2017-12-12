import os

import numpy
import pandas

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


def transform_ratings_into_classes(ratings):
    num_rows = ratings.shape[0]
    t = 2 * ratings - 1
    t = t.astype('int32')
    b = numpy.zeros((num_rows, 10))
    b[numpy.arange(num_rows), t] = 1
    return b

if __name__ == "__main__":
    train_users, train_movies, train_ratings = get_ratings_data()
    from IPython import embed
    embed()
