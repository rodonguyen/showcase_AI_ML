#
# Utility functions for CAB420, Assignment 1C, Q1
# Author: Simon Denman (s.denman@qut.edu.au)
#

import pandas   # to load csvs and manipulate the resulting data frames
import math     # for isnan
import os       # for path stuff
import pickle

def savedata(data, filename):
    with open(f"{filename}", "wb") as f:
        pickle.dump(data, f)


def loaddata(filename):
    with open(f"{filename}", "rb") as f:
        return pickle.load(f)


# load the data, in particular the movies and ratings files
#   base_path: directory that contains the quetions CSV files
#
#   returns:   dataframes for movies.csv and ratings.csv
#
def load_data(base_path):
    # simply using pandas.read_csv here
    movies = pandas.read_csv(os.path.join(base_path, 'movies.csv'))
    ratings = pandas.read_csv(os.path.join(base_path, 'ratings.csv'))
    return movies, ratings

# get a modified version of the ratings table, that has the average rating fo each film
#   ratings_table: data frame for the movies ratings
#
#   returns:       dataframe that contains movie IDs and the average ratings for each movie
#
def get_average_rating_per_film(ratings_table):
    # drop userId and timestamp as the average of these is meaningless
    # group the data by the movieId, and average the remaining columns (rating)
    return ratings_table.drop(columns=['userId', 'timestamp']).groupby('movieId').mean()

# replace the 'genre' column with a series of columns, one per genre, where a value of 1 
# indicates that the genre is presnet in the movie. Any genre not present will be set 
# to NaN
#   movies_table: dataframe for the movies
#
#   returns:      modified dataframe, where the genre column has been expanded to have one column for each genre, and the
#                 set of genres
#
def expand_genres(movies_table):

    # copy the table to create a working copy
    movies = movies_table.copy()
    # build a set of all the different genres, start with an empty set
    genres = set()
    # loop through all the movies
    for i, row in movies.iterrows():
        # build a set of genres for each movie, and get the union of this and the overall genre set
        genres = genres.union(set(row['genres'].split('|')))

    # create a column for each of our new genres
    for g in genres:
        movies[g] = float("Nan")

    # loop through the movies
    for i, row in movies.iterrows():
        # and the genres present in each movie
        for g in set(row['genres'].split('|')):
            # and where a genre is present in a movie, set it to 1
            movies.loc[i,g] = 1

    # drop the original genres column
    movies = movies.drop(columns=['genres'])
    
    # return expanded table
    return movies, genres

# build a table that has the details of all movies a user has seen. This will essentially replace the movieID with
# the genre flags for a film, and change the 1.0's that indicate what genre a movie is to the user rating for the film.
# This representation is primarily intended as an intermediate step to getting an average rating per user across
# genres, but you may use this (or vary it to get other aggregrated data) if you so choose.
#   ratings_table: dataframe of ratings
#   movies_table:  dataframe of movies, expanded to have individual columns for each genres
#   genres:        set of genres
#
#   returns:       merged dataframe, consisting of user IDs and the genres of all movies they've watched and the rating
#
def movies_per_user(ratings_table, movies_table, genres):
    # merge of movieId
    merged = pandas.merge(ratings_table, movies_table, how='left', on="movieId")
    # loop across the merged table
    for i, row in merged.iterrows():
        # and the set of genres
        for g in genres:
            # and for any genre that is set for a movie
            if not math.isnan(row[g]):
                # set it's value to the rating the user gave the movie
                merged.loc[i, g] = row["rating"]

    # drop the movieId, rating, timestamp, and title columns
    merged_all_movies = merged.drop(columns=['movieId', 'rating', 'timestamp', 'title'])                

    # return merged table
    return merged_all_movies

# get the average rating per genre for each user
#   movies_per_user_table: dataframe, that's the output of movies_per_user()
#
#   returns:               dataframe with one row per user, which contains the average rating given to movies of each
#                          genre. Genres that the user has never seen will have a value of NaN
#
def average_per_user(movies_per_user_table):
    return movies_per_user_table.groupby(['userId']).mean()
