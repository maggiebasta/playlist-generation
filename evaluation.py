import numpy as np
import pickle
import sys
from feature_extraction import compute_df_features

sys.path.append('Models')
import wrmf_helpers
from settings import PATH_TO_SPARSE_MATRIX

with open('data/wrmf_factors.pickle', 'rb') as f:
    playlist_factors, song_factors = pickle.load(f)

_, tid_to_idx, _, _, _ = wrmf_helpers.get_user_item_sparse_matrix(PATH_TO_SPARSE_MATRIX)


def eval_clicks(true_songs, predicted_songs, seeds):
    """
    computes the number of "clicks" for predictions to recommend
    a song from the true playlist. Each "click" recommends 10 songs

    :param true_songs: track_ids of the actual songs from the playlist
    :param predicted_songs: track_ids of the (ordered) model recommendations
    :param seeds: track_ids of the seed songs used as input
    :returns: number of clicks to see the first true song
    """
    n_preds = len(predicted_songs)
    true_set = set(true_songs)
    for i, idx in enumerate(np.arange(0, n_preds, 10)):
        preds = predicted_songs[idx:idx+10]
        for p in preds:
            if p in true_set and p not in seeds:
                return i
    return float("inf")


def get_model_clickscore(playlist, model):
    playlist = playlist.str.replace('spotify:track:', '')
    playlist_set = set(playlist)
    seed_ids = []
    while len(seed_ids) < 2:
        rand = list(playlist.sample(n=1))[0]
        if rand in tid_to_idx and rand not in seed_ids:
            seed_ids.append(rand)
    playlist_set.remove(seed_ids[0])
    playlist_set.remove(seed_ids[1])
    wrmf_ensemble_output = wrmf_helpers.get_top_similar_from_ensemble(
            song_factors,
            playlist_factors,
            seed_ids,
            n_similar_songs=10000,
            n_similar_playlists=100
    )
    wrmf_ensemble_output_set = set(wrmf_ensemble_output)

    # dummy column for compute_df_features function
    Y = [0]*len(wrmf_ensemble_output_set)
    X = compute_df_features(
        seed_ids, wrmf_ensemble_output, Y
    ).drop('relevence', axis=1)

    pred_probs = model.predict_proba(X)[:, 1]
    sorted_preds, _ = zip(*sorted(zip(X.index, pred_probs), key=lambda x: x[1]))
    return eval_clicks(playlist, sorted_preds, seed_ids)
