import json
import os
import sys

import implicit
import numpy as np
import pandas as pd
import pickle as pkl
import scipy.sparse as sparse

sys.path.append('../Models')
sys.path.append('../')
from spotify_api_database import SpotifyAuth, Track, fetchTracks
from Models.settings import (
    PATH_TO_SPARSE_MATRIX,
    PATH_TO_MDP_DATA_FOLDER
)


def get_user_item_sparse_matrix(path_to_matrix):
    """
    Retrieves sparse matrix for Playlist-track matrix implementation
    of User-item matrix. Assumes matrix stored in path_to_matrix

    :param path_to_matrix: path to sparse matrix file
    :returns: matrix and index and key lookups for songs and playlists
    """
    with open(path_to_matrix, 'rb') as f:
        matrix, tid_2_idx, idx_2_tid, pid_2_idx, idx_2_pid = pkl.load(f)

    return matrix, tid_2_idx, idx_2_tid, pid_2_idx, idx_2_pid


def get_song_name(tid):
    """
    Given an spotify track id, returns the name (for interpreting results)
    :param tid: spotify track id
    :returns: song name
    """
    spotify = SpotifyAuth()
    params = {'ids': [tid]}
    name = spotify._get(
        "https://api.spotify.com/v1/tracks", params
    )['tracks'][0]['name']
    return name


def get_fitted_wrmf(matrix_path, params):
    """
    Given the user-item sparse matrix and a dictionary of hyperparameters
    returns the WRMF top k predictions
    :param matrix: the user item matrix in scipy sparse form
    :param: hyperparameters to fit
    :params k: number of top songs to return for each song
    """
    if os.path.isfile('wrmf_playlist_factors.pickle'):
        raise FileExistsError("factors already saved locally")

    matrix, tid_to_idx, idx_to_tid, _, _ = get_user_item_sparse_matrix(
        PATH_TO_SPARSE_MATRIX
    )
    model = implicit.approximate_als.NMSLibAlternatingLeastSquares(
        factors=params['factors'],
        regularization=params['reg'],
        iterations=params['iters']
    )
    config = (matrix * params['alpha']).astype('double')
    model.fit(config)

    with open('wrmf_playlist_factors.pickle', 'wb') as fd:
        pkl.dump(model.item_factors, fd)
    with open('wrmf_song_factors.pickle', 'wb') as fd:
        pkl.dump(model.user_factors, fd)

    return model.item_factors, model.user_factors


def get_top_tracks(song_factors, track_id, n_similar, verbose=True):
    """
    Given a track id and a number of tracks to return, returns the
    n most similar tracks computer by implicit_mf.implicit_als_cg()
    :param top_k_matrix: matrix of top songs for each song (stage 1 output)
    :param n_similar: the number of recommendations to return
    :returns: list of n_similar tuples of track_ids and scores
    :verbose: boolean, whether or not to print results
    """

    # get conversions between index and spotify track id
    _, tid_to_idx, idx_to_tid, _, _ = get_user_item_sparse_matrix(
        PATH_TO_SPARSE_MATRIX
    )
    tidx = tid_to_idx[track_id]

    item_vecs = song_factors
    item_norms = np.sqrt((item_vecs * item_vecs).sum(axis=1))

    scores = item_vecs.dot(item_vecs[tidx]) / item_norms
    top_idx = np.argpartition(scores, -n_similar)[-n_similar:]
    similar = sorted(
        zip(top_idx, scores[top_idx] / item_norms[tidx]),
        key=lambda x: -x[1]
    )

    # Build return and print the names and scores of most similar
    ret = [idx_to_tid[idx] for idx, _ in similar]
    if verbose:
        print(f"\nRecommended Songs for {get_song_name(track_id)}")
        print('-' * 60)
        print('{:<50s}{:>4s}'.format("Track Name", "Score"))
        print('-' * 60)
        for idx, score in similar:
            # Look up track id
            tid = idx_to_tid[idx]

            # print track name and score
            name = get_song_name(tid)
            print('{:<50s}{:>4f}'.format(name, score))
    return ret


if __name__ == "__main__":
    params = {
        'factors': 20,
        'reg': 0.1,
        'iters': 20,
        'alpha': 15
    }
    get_fitted_wrmf(PATH_TO_SPARSE_MATRIX, params)
    # track_id = '1lzr43nnXAijIGYnCT8M8H'  # It Wasn't Me, by Shaggy
    # n_similar = 10
    # get_top_tracks(track_id, n_similar)
