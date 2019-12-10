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


def get_wrmf_factors(matrix_path, params):
    """
    Given the user-item sparse matrix and a dictionary of hyperparameters
    returns the WRMF top k predictions
    :param matrix: the user item matrix in scipy sparse form
    :param: hyperparameters to fit
    :returns: tuple of playlist (item), and song (user) factors
    """
    if os.path.isfile('wrmf_factors.pickle'):
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

    with open('wrmf_factors.pickle', 'wb') as fd:
        pkl.dump((model.item_factors, model.user_factors), fd)

    return model.item_factors, model.user_factors


def get_top_similar_from_tracks(
    song_factors,
    track_ids,
    n_similar,
    verbose=True
):
    """
    Given seed tracks and a number of tracks to return, returns the
    n most similar tracks computed from wrmf
    :param song_factors: song_factors from wrmf
    :param track_ids: the ids of the seed tracks
    :param n_similar: number of similar tracks to return
    :param verbose: whether or not to print results
    :returns: list of n_similar tuples of track_ids and scores
    :verbose: boolean, whether or not to print results
    """

    # get conversions between index and spotify track id
    _, tid_to_idx, idx_to_tid, _, _ = get_user_item_sparse_matrix(
        PATH_TO_SPARSE_MATRIX
    )
    tidxs = [tid_to_idx[tid] for tid in track_ids]

    item_vecs = song_factors
    item_norms = np.sqrt((item_vecs * item_vecs).sum(axis=1))

    scores = np.sum(item_vecs.dot(item_vecs[tidxs].T), axis=1) / item_norms
    top_idx = np.argpartition(scores, -n_similar)[-n_similar:]
    norm = sum([item_norms[tidx] for tidx in tidxs])
    similar = sorted(
        zip(top_idx, scores[top_idx]/norm),
        key=lambda x: -x[1]
    )

    # Build return and print the names and scores of most similar
    ret = [idx_to_tid[idx] for idx, _ in similar]
    if verbose:
        seed_names = [get_song_name(tid) for tid in track_ids]
        print(f"\nRecommended Songs for {seed_names}")
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


def get_top_similar_from_playlists(
    song_factors,
    playlist_factors,
    track_ids,
    n_similar_playlists,
    n_similar_songs,
    verbose=True
):
    """
    Given seed tracks and a number of playlists to return, returns the
    n most similar playlists computed by wrmf
    :param song_factors: song_factors from wrmf
    :param playlist_factors: playlist_factors from wrmf
    :param track_ids: the ids of the seed tracks
    :param n_similar: number of similar tracks to return
    :param verbose: whether or not to print results
    :returns: list of n_similar tuples of track_ids and scores
    :verbose: boolean, whether or not to print results
    """

    # get conversions between index and spotify track id
    matrix, tid_to_idx, idx_to_tid, _, _ = get_user_item_sparse_matrix(
        PATH_TO_SPARSE_MATRIX
    )
    tidxs = [tid_to_idx[tid] for tid in track_ids]

    item_vecs = song_factors

    scores = np.sum(item_vecs[tidxs].dot(playlist_factors.T), axis=0)
    top_idx = np.argpartition(
        scores, -n_similar_playlists
    )[-n_similar_playlists:]
    similar_playlists, _ = zip(*sorted(
        zip(top_idx, scores[top_idx]),
        key=lambda x: -x[1]
    ))

    top_songs_idxs = np.argsort(
        np.sum(matrix[np.array(similar_playlists)], axis=0)
    )
    top_songs_idxs = np.array(top_songs_idxs).flatten()[-n_similar_songs:]
    return [idx_to_tid[idx] for idx in top_songs_idxs]


def get_top_similar_from_ensemble(
    song_factors,
    playlist_factors,
    track_ids,
    n_similar_playlists,
    n_similar_songs,
    verbose=True
):
    """
    Given seed tracks and a number of playlists to return, returns the
    n most similar playlists computed by wrmf
    :param song_factors: song_factors from wrmf
    :param playlist_factors: playlist_factors from wrmf
    :param track_ids: the ids of the seed tracks
    :param n_similar: number of similar tracks to return
    :param verbose: whether or not to print results
    :returns: list of n_similar tuples of track_ids and scores
    :verbose: boolean, whether or not to print results
    """

    # get conversions between index and spotify track id
    matrix, tid_to_idx, idx_to_tid, _, _ = get_user_item_sparse_matrix(
        PATH_TO_SPARSE_MATRIX
    )
    tidxs = [tid_to_idx[tid] for tid in track_ids]

    item_vecs = song_factors

    song_song_scores = np.sum(item_vecs.dot(item_vecs[tidxs].T), axis=1)

    playlist_scores = np.sum(item_vecs[tidxs].dot(playlist_factors.T), axis=0)
    top_idx = np.argpartition(
        playlist_scores, -n_similar_playlists
    )[-n_similar_playlists:]
    similar_playlists, _ = zip(*sorted(
        zip(top_idx, playlist_scores[top_idx]),
        key=lambda x: -x[1]
    ))
    playlist_song_scores = np.sum(matrix[np.array(similar_playlists)], axis=0)

    scores = playlist_song_scores + song_song_scores
    top_songs_idxs = np.argsort(scores)
    top_songs_idxs = np.array(top_songs_idxs).flatten()[-n_similar_songs:]
    return [idx_to_tid[idx] for idx in top_songs_idxs]


if __name__ == "__main__":
    params = {
        'factors': 20,
        'reg': 0.1,
        'iters': 20,
        'alpha': 15
    }
    get_wrmf_factors(PATH_TO_SPARSE_MATRIX, params)
