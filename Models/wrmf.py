import pandas as pd
import numpy as np
import seaborn as sns
import scipy.sparse as sparse
from sklearn.metrics import mean_squared_error
import os
import sys
import pickle

# Needed for Jupyter notebook Sublime integration
PATH_TO_SCRIPT = '/Users/mwornow/Desktop/Dropbox/School/Stat121/Project/playlist-generation/Models/'
sys.path.insert(0, PATH_TO_SCRIPT)
PATH_TO_SCRIPT = '/Users/mwornow/Desktop/Dropbox/School/Stat121/Project/playlist-generation/'
sys.path.insert(0, PATH_TO_SCRIPT)
from implicit_mf import implicit_als_cg
from gen_spotify_api_database import Track, fetchTracks

# Constants
PATH_TO_SPARSE_MATRIX = PATH_TO_SCRIPT + 'Data/mdp_wrmf_sparse_matrix.pickle'
PATH_TO_MDP_DATA_FOLDER = '/Users/mwornow/desktop/Stat121Data/'

# User-item ==> Playlist-track

def get_user_item_sparse_matrix(path_to_matrix):
	if os.path.isfile(path_to_matrix):
		with open(path_to_matrix, 'rb') as fd:
			matrix, tid_to_idx, idx_to_tid, pid_to_idx, idx_to_pid = pickle.load(fd)
	else:
		matrix, tid_to_idx, idx_to_tid, pid_to_idx, idx_to_pid = read_all_csvs()
		with open(path_to_matrix, 'wb') as fd:
			pickle.dump((matrix, tid_to_idx, idx_to_tid, pid_to_idx, idx_to_pid), fd)
	return matrix, tid_to_idx, idx_to_tid, pid_to_idx, idx_to_pid

def read_all_csvs():
	########
	# (1) Read all "songs1.csv" files from MPD, store in Pandas Dataframe with Track URI and Playlist ID
	#######
	dfs = []
	cols_ignore = ['pos', 'artist_name', 'artist_uri', 'track_name', 'album_uri', 'duration_ms', 'album_name']
	for i in range(1000):
		filename = PATH_TO_MDP_DATA_FOLDER + 'songs' + str(i) + '.csv'
		df = pd.read_csv(filename).drop(columns = cols_ignore)
		df['pid'] = df['pid'] + i * 1000
		dfs.append(df)
		print("Done reading "+filename)
		if i > 4:
			break
	df = pd.concat(dfs)
	#####
	# (2) Convert Track URIs -> Track IDs, drop Track URI
	####
	df['tid'] = df['track_uri'].str.replace('spotify:track:', '')
	df = df.drop(columns = [ 'track_uri' ])
	# Now df is a 2-column DataFrame with:
	# pid			|	tid
	# 11999	  		| 	2ovm5VZJ36s3HKF8nkcIZI
	print("# of Playlist-Track pairs: ", df.shape[0])
	#####
	# (3) Convert DataFrame -> Sparse Matrix (# of playlists x # of tracks), # of total entries (nnz) should equal df.shape[0]
	####
	# Create mappings
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

if __name__ == 'main':
	matrix, tid_to_idx, idx_to_tid, pid_to_idx, idx_to_pid = get_user_item_sparse_matrix(PATH_TO_SPARSE_MATRIX)
	print("Sparse Matrix Dims:", matrix.shape, "| # of Entries in Sparse Matrix:", matrix.nnz)

	alpha_val = 15
	conf_data = (matrix * alpha_val).astype('double')
	playlist_vecs, track_vecs = implicit_als_cg(conf_data, iterations=2, features=20)

	print("Fitted Feature Vectors:", playlist_vecs.shape, track_vecs.shape)

	# Find the 10 most similar to Jay-Z
	track_id = '1lzr43nnXAijIGYnCT8M8H' # It Wasn't Me, by Shaggy
	tidx = tid_to_idx[track_id]
	track_vec = track_vecs[tidx].T
	n_similar = 10

	# Calculate the similarity score between Mr Carter and other artists
	# and select the top 10 most similar.
	scores = track_vecs.dot(track_vec).toarray().reshape(1,-1)[0]
	top_10 = np.argsort(scores)[::-1][:n_similar]

	# Get and print the actual artists names and scores
	for idx in top_10:
		print(idx_to_tid[idx])


	# Calculate the vector norms
	track_norms = sparse.linalg.norm(track_vecs, axis = 1, ord = 2) * sparse.linalg.norm(track_vec)

	# Calculate the similarity score, grab the top N items and
	# create a list of item-score tuples of most similar tracks
	track_vec = track_vecs[tidx].T
	scores = track_vecs.dot(track_vec) / track_norms # Cosine similarity instead of dot products
	top_idx = np.argpartition(scores, -n_similar)[-n_similar:]
	similar = sorted(zip(top_idx, scores[top_idx]), key=lambda x: -x[1])

	# Print the names of our most similar tracks
	print(similar)

