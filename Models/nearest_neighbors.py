import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import sys
import os

from wrmf import get_user_item_sparse_matrix
from gen_spotify_api_database import Track, fetchTracks, Spotify, SpotifyAuth

# Constants
PATH_TO_SPARSE_MATRIX = '../Data/mdp_wrmf_sparse_matrix.pickle'

def kneighbors_fit(t_by_p_matrix):
	model_kneighbors = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
	model_kneighbors.fit(t_by_p_matrix)
	return model_kneighbors
def kneighbors_predict(model_kneighbors, t_by_p_matrix, track_id, n_similar = 10):
	# Returns top `n_similar` tracks most similar to `track_id`
	## Map track_id -> Index in t_by_p_matrix
	query_index = tid_to_idx[track_id]
	distances, indices = model_kneighbors.kneighbors(t_by_p_matrix[query_index, :].reshape(1, -1), n_neighbors = n_similar + 1)
	## Flatten np.array()'s
	indices = indices.flatten()
	distances = distances.flatten()
	## Throw out trivial neighbor of track_id itself if included in returned neighbors
	if query_index == indices[0]:
		indices = indices[1:]
		distances = distances[1:]
	else:
		indices = indices[:-1]
		distances = distances[:-1]
	return [ idx_to_tid[idx] for idx in indices ], distances
def kneighbors_get_similar_tracks(model_kneighbors, t_by_p_matrix, track_id):
	# Make K-NN predictions
	kneighbors_preds, kneighbors_dists = kneighbors_predict(model_kneighbors, t_by_p_matrix, track_id, n_similar = 10)
	# Query Spotify to interpret our results
	sp = SpotifyAuth()
	tracks = fetchTracks(sp, [track_id] + kneighbors_preds, verbose = False)
	# Print out results
	print("Recommendations for: "+tracks[0].name+", by " + tracks[0].get_artists() + " (ID: " + str(tracks[0].id) + ")")
	for i in range(1, len(tracks)):
		# Skip first track in tracks b/c it's the track we're making recs for
		print("#" + str(i) + ": " + tracks[i].name + ", by " + tracks[i].get_artists() + " (ID: " + str(tracks[i].id) + ", Dist: " + str(kneighbors_dists[i-1]) + ")")

if __name__ == '__main__':
	# Read sparse matrix
	p_by_t_matrix, tid_to_idx, idx_to_tid, pid_to_idx, idx_to_pid = get_user_item_sparse_matrix(PATH_TO_SPARSE_MATRIX)
	print("Sparse Matrix Dims:", p_by_t_matrix.shape, "| # of Entries in Sparse Matrix:", p_by_t_matrix.nnz)
	# Convert Playlist x Track matrix --> Track x Playlist
	t_by_p_matrix = p_by_t_matrix.transpose()
	# Fit Nearest Neighbors model
	model_kneighbors = kneighbors_fit(t_by_p_matrix)
	track_id = '1lzr43nnXAijIGYnCT8M8H'
	kneighbors_get_similar_tracks(model_kneighbors, t_by_p_matrix, track_id)