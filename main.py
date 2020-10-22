import pandas as pd
import operator
import numpy as np

from csv import reader
from scipy import spatial
from scipy.sparse.linalg import svds

print("Preparing System...")
df_ratings = pd.read_csv('ml-20m/ratings.csv',
                         usecols=['userId', 'movieId', 'rating'],
                         dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'}, encoding='utf-8')

movieProperties = df_ratings.groupby('movieId').agg({'rating': [np.size, np.mean]})
print("loading Data...")
# mf

df_ratings_new = df_ratings[:2000000]
df_movie_features = df_ratings_new.pivot(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)

R = df_movie_features.to_numpy()
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

M, k, N = svds(R_demeaned, k=50)
k = np.diag(k)
print("Factorizing Matrix...")

movieNumRatings = pd.DataFrame(movieProperties['rating']['size'])
movieNormalizedNumRatings = movieNumRatings.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

all_genres = ['(no genres listed)', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
              'Drama', 'Fantasy',
              'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', 'IMAX']

movieDict = {}

filename = 'ml-20m/movies.csv'

with open(filename, 'r', encoding='utf-8') as f:
    next(f)
    csv_reader = reader(f)
    for row in csv_reader:

        movieId = int(row[0])
        name = row[1]
        genres = row[2].split('|')
        list_genre = [0 for _ in range(0, 20)]
        for genre in genres:
            list_genre[all_genres.index(genre)] = 1
        try:
            movieNormalizedNumRatings.loc[movieId].get('size')
            check = True
        except KeyError:

            check = False
        if (check):
            movieDict[movieId] = (name, np.array(list_genre), movieNormalizedNumRatings.loc[movieId].get('size'),
                                  movieProperties.loc[movieId].rating.get('mean'))
        else:
            pass
            # movieDict[3] = (
            #     'Grumpier Old Men (1995)', np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
            #     movieNormalizedNumRatings.loc[3].get('size'),
            #     movieProperties.loc[3].rating.get('mean'))


def l2_distance(row1, row2):
    # l2 no loop
    genresA = row1[1]
    genresB = row2[1]
    genreDistance = spatial.distance.cosine(genresA, genresB)

    popularityA = row1[2:]
    popularityB = row2[2:]
    num_test = popularityA.shape[0]
    num_train = popularityB.shape[0]
    dists = np.zeros((num_test, num_train))
    dists = np.sqrt(
        (popularityB * popularityB).sum(axis=None) + (popularityA * popularityA).sum(axis=None) - 2 * popularityB.dot(
            popularityA.T))
    return genreDistance + dists


def getNeighbors(movieID, K):
    distances = []
    for movie in movieDict:
        if (movie != movieID):
            dist = l2_distance(np.array(movieDict[movieID], dtype=object), np.array(movieDict[movie], dtype=object))
            distances.append((movie, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(K):
        neighbors.append(distances[x][0])
    return neighbors


K = 10
avgRating = 0

i = 0;
movie = str(input("Type your fav movie!"))

print("Do you mean? Select:")

j = 0
sim_dict = {}
for key, val in movieDict.items():
    if movie.lower() in val[0].lower():
        j = j + 1
        sim_dict[j] = val[0]

        print("{}. {}?".format(j, val[0]))

        # i = key
        #
print("Select movie:")
id = int(input())

for key, val in sim_dict.items():

    if key == id:
        movie = val
        break
# final = -1
print(movie)
for key, val in movieDict.items():
    if movie.lower() == val[0].lower():
        print('You Selected: {}'.format(val[0]))
        id = key;

neighbors = getNeighbors(id, K)
print("Movies Recommended:")
for neighbor in neighbors:
    avgRating += movieDict[neighbor][3]
    print(movieDict[neighbor][0] + " " + str(movieDict[neighbor][3]))
