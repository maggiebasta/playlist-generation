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
from gen_spotify_api_database import SpotifyAuth, Track, fetchTracks
from settings import (
    PATH_TO_SPARSE_MATRIX,
    PATH_TO_MDP_DATA_FOLDER
)


def get_user_item_sparse_matrix(path_to_matrix):
    """
    Retrieves sparse matrix for Playlist-track matrix implementation
    of User-item matrix

    :param path_to_matrix: path to sparse matrix file
    :returns: matrix and index and key lookups for songs and playlists
    """

    if os.path.isfile(path_to_matrix):
        with open(path_to_matrix, 'rb') as f:
            matrix, tid_2_idx, idx_2_tid, pid_2_idx, idx_2_pid = pkl.load(f)
    else:
        matrix, tid_2_idx, idx_2_tid, pid_2_idx, idx_2_pid = read_all_csvs()
        with open(path_to_matrix, 'wb') as f:
            pkl.dump((matrix, tid_2_idx, idx_2_tid, pid_2_idx, idx_2_pid), f)

    return matrix, tid_2_idx, idx_2_tid, pid_2_idx, idx_2_pid


def read_all_csvs():
    """
    Creates sparse matrix for Playlist-track matrix implementation
    of User-item matrix using the MDP slices

    :returns: matrix and index and key lookups for songs and playlists
    """
    ########
    # (1) Read all MPD csvs, store in Dataframe w/ Track URI and Playlist ID
    ########

    dfs = []
    cols_ignore = [
        'pos',
        'artist_name',
        'artist_uri',
        'track_name',
        'album_uri',
        'duration_ms',
        'album_name'
    ]
    for i in range(1000):
        filename = PATH_TO_MDP_DATA_FOLDER + 'songs' + str(i) + '.csv'
        df = pd.read_csv(filename).drop(columns=cols_ignore)
        df['pid'] = df['pid'] + i * 1000
        dfs.append(df)
        print("Done reading "+filename)
        if i > 4:
            break
        df = pd.concat(dfs)

    ########
    # (2) Convert Track URIs -> Track IDs, drop Track URI
    # df is a 2-column DataFrame with:
    # pid		|	tid
    # 11999	  	| 	2ovm5VZJ36s3HKF8nkcIZI
    # ...
    #########

    df['tid'] = df['track_uri'].str.replace('spotify:track:', '')
    df = df.drop(columns=['track_uri'])

    print("# of Playlist-Track pairs: ", df.shape[0])

    ########
    # (3) Convert DataFrame -> (#playlists X #tracks) Sparse Matrix,
    # total entries should = df.shape[0]
    ########

    # Create id to index mappings
    tid_to_idx = {}
    idx_to_tid = {}
    for (idx, tid) in enumerate(df['tid'].unique().tolist()):
        tid_to_idx[tid] = idx
        idx_to_tid[idx] = tid
    pid_to_idx = {}
    idx_to_pid = {}
    for (idx, pid) in enumerate(df['pid'].unique().tolist()):
        pid_to_idx[pid] = idx
        idx_to_pid[idx] = pid

    # Convert DataFrame -> Sparse Matrix
    def map_ids(row, mapper):
        return mapper[row]

    I = df['pid'].apply(map_ids, args=[pid_to_idx]).to_numpy()
    J = df['tid'].apply(map_ids, args=[tid_to_idx]).to_numpy()
    V = np.ones(I.shape[0])

    matrix = sparse.coo_matrix((V, (I, J)), dtype=np.float64)
    matrix = matrix.tocsr()
    return matrix, tid_to_idx, idx_to_tid, pid_to_idx, idx_to_pid


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


def get_top_tracks(track_id, n_similar):
    """
    Given a track id and a number of tracks to return, returns the
    n most similar tracks computer by implicit_mf.implicit_als_cg()
    :param track_id: ID of song to match
    :param n_similar: the number of recommendations to return
    :returns: list of n_similar tuples of track_ids and scores
    """

    matrix, tid_to_idx, idx_to_tid, _, _ = get_user_item_sparse_matrix(
        PATH_TO_SPARSE_MATRIX
    )
    print(
        f"Sparse Matrix Dims:{matrix.shape} | "
        "# of Entries in Sparse Matrix:{matrix.nnz}"
    )
    model = implicit.als.AlternatingLeastSquares(
        factors=20,
        regularization=0.1,
        iterations=20
    )
    alpha_val = 15
    data_conf = (matrix * alpha_val).astype('double')
    model.fit(data_conf)

    item_vecs = model.item_factors

    # Calculate the vector norms
    item_norms = np.sqrt((item_vecs * item_vecs).sum(axis=1))

    # Calculate the similarity score, grab the top N items and
    # create a list of item-score tuples of most similar artists
    tidx = tid_to_idx[track_id]

    scores = item_vecs.dot(item_vecs[tidx]) / item_norms
    top_idx = np.argpartition(scores, -n_similar)[-n_similar:]
    similar = sorted(
        zip(top_idx, scores[top_idx] / item_norms[tidx]),
        key=lambda x: -x[1]
    )

    # Build return and print the names and scores of most similar
    ret = []
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

        # append track id and score to return
        ret.append([tid, score])

    return ret

if __name__ == "__main__":
    track_id = '1lzr43nnXAijIGYnCT8M8H'  # It Wasn't Me, by Shaggy
    n_similar = 10
    get_top_tracks(track_id, n_similar)
