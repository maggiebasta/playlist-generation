import itertools
import pickle
import sys
from time import time

import pandas as pd
import numpy as np

sys.path.append('Models')
import wrmf_helpers
from settings import PATH_TO_SPARSE_MATRIX

_, tid_to_idx, _, _, _ = wrmf_helpers.get_user_item_sparse_matrix(PATH_TO_SPARSE_MATRIX)


from spotify_api_database import Track
INPUT_FILE = '/Users/mabasta/Desktop/CS109a/playlist-generation/data/spotify_dictionary.pickle'
with open(INPUT_FILE, 'rb') as fd:
	SpotifyData = pickle.load(fd)


TrackIDs = SpotifyData.keys()
N_TRACKS = len(TrackIDs)


def get_song_features(tid):
    """
    Given an spotify track id, returns the audio features from the api
    :param tid: spotify track id
    :returns: song features
    """

    # dictionary of features to return
    spotify_track_data = SpotifyData[tid]

    features = {}
    features['name'] = spotify_track_data.name
    features['artists'] = spotify_track_data.artists
    features['popularity'] = spotify_track_data.popularity
    features['album'] = spotify_track_data.album_name
    features['danceability'] = spotify_track_data.danceability
    features['energy'] = spotify_track_data.energy
    features['key'] = spotify_track_data.key
    features['loudness'] = spotify_track_data.loudness
    features['mode'] = spotify_track_data.mode
    features['speechiness'] = spotify_track_data.speechiness
    features['acousticness'] = spotify_track_data.acousticness
    features['instrumentalness'] = spotify_track_data.instrumentalness
    features['liveness'] = spotify_track_data.liveness
    features['valence'] = spotify_track_data.valence
    features['tempo'] = spotify_track_data.tempo
    features['duration_ms'] = spotify_track_data.duration_ms
    features['time_signature'] = spotify_track_data.time_signature

    return features


def get_features_dataframe(tids):
    """
    Given a list of spotify track ids returns a dataframe of the
    audio features for the tracks
    :param tids: list of spotify track ids
    :returns: dataframe of song features
    """

    Data = {}
    for tid in tids:
        Data[tid] = get_song_features(tid)
    return pd.DataFrame.from_dict(Data, orient='index')


def compute_seedset_features(tids):
    """
    Given a list of seed track ids returns a dataframe summarizing
    average or cumualitve features for the seed playlist
    :param tids: list of spotify track ids
    :returns: dictionary of summarizing features
    """
    seed_dataframe = get_features_dataframe(tids)
    summary_feats = {}
    summary_feats['names'] = list(seed_dataframe['name'])
    summary_feats['artists'] = list(
        itertools.chain.from_iterable(seed_dataframe['artists'])
    )
    summary_feats['albums'] = list(seed_dataframe['album'])
    numeric = [
        'popularity', 'danceability', 'energy', 'key', 'loudness',
        'mode', 'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature'
    ]
    for feat in numeric:
        summary_feats[feat] = seed_dataframe[feat].dropna().mean()
    return summary_feats


def compute_df_features(seed_tids, candidate_tids, relevences):
    """
    Given a list of seed track ids and candidate_tids returns a
    dataframe summarizing the differences between each candidate
    track's features and seed features.
    :param seed_tids: seed track ids
    :param candidate_tids: candidate track ids
    :param relevences: list of relevence for each candidate
    """
    seed_features = compute_seedset_features(seed_tids)

    # drop candidate songs w/0 all features
    candidate_df = get_features_dataframe(candidate_tids)
    candidate_df['relevence'] = relevences
    candidate_df.dropna(axis=0)
    df = {}
    df['relevence'] = candidate_df['relevence']
    df['artist_overlap'] = [
        1 if len(np.intersect1d(x, seed_features['artists'])) else 0
        for x in candidate_df['artists']
    ]
    df['album_overlap'] = [
        1 if x in seed_features['albums'] else 0
        for x in candidate_df['album']
    ]

    numeric = [
        'popularity', 'danceability', 'energy', 'key', 'loudness',
        'mode', 'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature'
    ]
    for feat in numeric:
        diff = (candidate_df[feat] - seed_features[feat]).abs()
        df[f'{feat}_diff'] = diff

    df = pd.DataFrame.from_dict(df)
    return df


def compute_df(playlist, song_factors, playlist_factors):
    """
    Given an input playlist and factors computed from stage 1,
    returns a df for stage 2

    :param playlist: list of song ids making up the playlist
    :param song_factors: song factors from wrmf
    :param playlist_factors: playlist factors from wrmf
    """
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
    true_matches = playlist_set.intersection(wrmf_ensemble_output_set)
    false_matches = wrmf_ensemble_output_set.symmetric_difference(true_matches)

    X_train_ids = []
    Y_train = []
    for _ in range(min(len(true_matches), 10)):
        X_train_ids.append(true_matches.pop())
        Y_train.append(1)
        X_train_ids.append(false_matches.pop())
        Y_train.append(0)

    return compute_df_features(seed_ids, X_train_ids, Y_train)
