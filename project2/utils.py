import numpy as np

def normalizeRatings(Y, R):

    m, n = Y.shape
    Ymean = np.zeros(m)
    Ynorm = np.zeros(Y.shape)

    for i in range(m):
        idx = R[i, :] == 1
        Ymean[i] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]

    return Ynorm, Ymean


def loadMovieList():
    """
    Đọc danh sách phim trong movie_ids.txt và trả về danh sách tên phim.
    """
    # Đọc danh sách phim 
    with open('movie_ids.txt',  encoding='ISO-8859-1') as fid:
        movies = fid.readlines()

    movieNames = []
    for movie in movies:
        parts = movie.split()
        movieNames.append(' '.join(parts[1:]).strip())
        
    return movieNames

