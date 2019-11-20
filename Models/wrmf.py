import implicit
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.sparse as sparse
from sklearn.metrics import mean_squared_error
import os
import sys
import pickle
import itertools

# Needed for Jupyter notebook Sublime integration
PATH_TO_SCRIPT = '/Users/mwornow/Desktop/Dropbox/School/Stat121/Project/playlist-generation/'
sys.path.insert(0, PATH_TO_SCRIPT)
from gen_spotify_api_database import Track, fetchTracks

# Constants
PATH_TO_SPARSE_MATRIX = PATH_TO_SCRIPT + 'Data/mdp_wrmf_sparse_matrix.pickle'
PATH_TO_MDP_DATA_FOLDER = '/Users/mwornow/desktop/Stat121Data/'
PATH_TO_WRMF_GRID_SEARCH_RESULTS = PATH_TO_SCRIPT + 'Data/mdp_wrmf_grid_search_results.pickle'

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
		if i > 1:
			break
	df = pd.concat(dfs).drop_duplicates() # Binarize dataset (some tracks appear >1 times per playlist)
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
	p_by_t_matrix = sparse.coo_matrix((V, (I, J)), dtype=np.float64)
	p_by_t_matrix = p_by_t_matrix.tocsr()
	return p_by_t_matrix, tid_to_idx, idx_to_tid, pid_to_idx, idx_to_pid

	def train_test_split(user_item_matrix, split_count = 5, fraction = 1):
		"""
		Split recommendation data into train and test sets
		
		Params
		------
		ratings : scipy.sparse matrix
			Interactions between users and items.
		split_count : int
			Number of user-item-interactions per user to move
			from training to test set.
		fractions : float
			Fraction of users to split off some of their
			interactions into test set. If None, then all 
			users are considered.
		"""
		if not (fraction >= 0 and fraction <= 1):
			fraction = 1
		# Note: likely not the fastest way to do things below.
		train = user_item_matrix.copy().tocoo()
		test = sparse.lil_matrix(train.shape)
		
		# (1) Get all rows with > split_count tracks marked as 1
		splittable_rows, _, _ = sparse.find(train.sum(axis = 1) > (split_count * 2))
		test_row_indices = np.random.choice(splittable_rows, size = fraction * splittable_rows.shape[0], replace = False)

		train = train.tolil()
		for test_row_idx in test_row_indices:
			try:
				test_cols = np.random.choice(user_item_matrix.getrow(test_row_idx).indices, 
												size = split_count, 
												replace=False)
				train[test_row_idx, test_cols] = 0
				test[test_row_idx, test_cols] = user_item_matrix[test_row_idx, test_cols]
			except Exception as e:
				print(str(e))
		
		# Test and training are truly disjoint
		assert(train.multiply(test).nnz == 0)
		return train.tocsr(), test.tocsr(), test_row_indices

if __name__ == '__main__':

	# Save Grid Search results
	with open(PATH_TO_WRMF_GRID_SEARCH_RESULTS, 'rb') as fd:
		obj = pickle.loads(fd.read())

	print(obj)
	exit()
	np.random.seed(10)
	# Read entire Playlist x Track matrix
	p_by_t_matrix, tid_to_idx, idx_to_tid, pid_to_idx, idx_to_pid = get_user_item_sparse_matrix(PATH_TO_SPARSE_MATRIX)
	print("Sparse Matrix Dims:", p_by_t_matrix.shape, "| # of Entries in Sparse Matrix:", p_by_t_matrix.nnz)

	# Transpose matrix into Track x Playlist (item_user) matrix as implicit expects
	t_by_p_matrix = p_by_t_matrix.transpose()

	train, test, test_row_indices = train_test_split(p_by_t_matrix, split_count = 5, fraction = 1)

	hyperparams = {
		'factors' : [10, 20, 30, 40, 50],
		'regularization' : [10, 1, 0, 0.1, 0.01, 0.001],
		'alpha' : [ 15 ],
	}

	N_ITERATIONS = 30
	results = []
	for v in itertools.product(*hyperparams.values()):
		params = dict(zip(hyperparams.keys(), v))
		train_conf = (train * params['alpha']).astype('double').transpose() # ALS() expects item_user matrix, not user_item, so need to transpose
		model_wrmf = implicit.approximate_als.NMSLibAlternatingLeastSquares(factors = params['factors'],
																			regularization = params['regularization'],
																			calculate_training_loss = True)
		train_errors = []
		test_errors = []
		for iteration in range(1, N_ITERATIONS + 1):
			model_wrmf.iterations = iteration
			model_wrmf.fit(train_conf)
			preds = model_wrmf.user_factors.dot(model_wrmf.item_factors.T) # This returns a Numpy array
			top_k_preds = np.argsort(preds[test_row_indices,:], axis = 1)[:,:500]
			r_test_precision = 0.0
			r_train_precision = 0.0
			for pred_idx in range(top_k_preds.shape[0]):
				test_true_indices = test.getrow(test_row_indices[pred_idx]).indices
				r_test_precision += len(set(top_k_preds[pred_idx,:test_true_indices.shape[0]]) & set(test_true_indices))/test_true_indices.shape[0] # Implement Spotify RecSys's R-precision score
				train_true_indices = train.getrow(test_row_indices[pred_idx]).indices
				r_train_precision += len(set(top_k_preds[pred_idx,:train_true_indices.shape[0]]) & set(train_true_indices))/train_true_indices.shape[0] # Implement Spotify RecSys's R-precision score
			r_test_precision /= top_k_preds.shape[0] # Convert sum of R-preds to mean
			r_train_precision /= top_k_preds.shape[0] # Convert sum of R-preds to mean
			test_errors.append(r_test_precision)
			train_errors.append(r_train_precision)
			print("R-Precision @ iteration "+str(iteration)+": " + str(r_test_precision))
		results.append({ 	'params' : params,
							'train_mse' : train_errors,
							'test_mse' : test_errors,
						})
	# Save Grid Search results
	with open(PATH_TO_WRMF_GRID_SEARCH_RESULTS, 'wb') as fd:
		pickle.dump(results, fd)
