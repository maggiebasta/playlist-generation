import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import sys
import os

from wrmf import get_user_item_sparse_matrix
from gen_spotify_api_database import Track, fetchTracks, Spotify, SpotifyAuth

# Constants
PATH_TO_SPARSE_MATRIX = '../Data/mdp_wrmf_sparse_matrix.pickle'

# He's trying to predict similar artists given user prefs --> He wants: artists x users
# We're trying to predict similar tracks given playlist prefs --> We want: tracks x playlists (but matrix is playlists x tracks)
matrix, tid_to_idx, idx_to_tid, pid_to_idx, idx_to_pid = get_user_item_sparse_matrix(PATH_TO_SPARSE_MATRIX)
print("Sparse Matrix Dims:", matrix.shape, "| # of Entries in Sparse Matrix:", matrix.nnz)

track_by_playlist_matrix = matrix.transpose()
model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(track_by_playlist_matrix)

query_index = tid_to_idx['1lzr43nnXAijIGYnCT8M8H']
distances, indices = model_knn.kneighbors(track_by_playlist_matrix[query_index, :].reshape(1, -1), n_neighbors = 6)


# Query Spotify to interpret our results
sp = SpotifyAuth()
main_track = fetchTracks(sp, [ idx_to_tid[query_index] ], verbose = False)[0]
tracks = fetchTracks(sp, [idx_to_tid[i] for i in indices.flatten()], verbose = False)

for i in range(0, len(distances.flatten())):
	if i == 0:
		print('Recommendations for {0}:\n'.format(main_track.name))
	else:
		print('{0}: {1}, with distance of {2}:'.format(i, tracks[i].name, distances.flatten()[i]))