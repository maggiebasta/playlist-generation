import numpy as np


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
