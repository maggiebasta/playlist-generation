import itertools

import pandas as pd
import numpy as np

from spotify_api_database import SpotifyAuth


def get_song_features(tid):
    """
    Given an spotify track id, returns the audio features from the api
    :param tid: spotify track id
    :returns: song features
    """
    spotify = SpotifyAuth()
    params = {'ids': [tid]}

    # dictionary of features to return
    features = {}

    # get name, artist, popularity, and album features
    spotify_features = spotify._get(
        "https://api.spotify.com/v1/tracks", params
    )['tracks'][0]

    features['name'] = spotify_features.get('name', "")
    features['artists'] = [
        artist['name'] for artist in spotify_features.get('artists', [])
    ]
    features['popularity'] = spotify_features.get('popularity', "")
    try:
        features['album'] = spotify_features['album']['name']
    except KeyError:
        features['album'] = ""

    # try to extract audio features
    spotify_audio_features = spotify._get(
        "https://api.spotify.com/v1/audio-features", params
    )['audio_features']

    audiofeats = [
        'danceability', 'energy', 'key', 'loudness',
        'mode', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence',
        'tempo', 'duration_ms', 'time_signature'
    ]
    if spotify_audio_features:
        for afeat in audiofeats:
            features[afeat] = spotify_audio_features[0][afeat]
    else:
        for afeat in audiofeats:
            features[afeat] = None

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


def get_X_train(seed_tids, candidate_tids):
    """
    Given a list of seed track ids and candidate_tids returns a
    dataframe summarizing the differences between each candidate
    track's features and seed features. Note, Xtrain should have
    nulls handled
    :seed_tids: seed track ids
    :returns: candidate_tids candidate track ids
    """
    seed_features = compute_seedset_features(seed_tids)

    # drop candidate songs w/0 all features
    candidate_df = get_features_dataframe(candidate_tids).dropna(axis=0)
    Xtrain = {}
    Xtrain['artist_overlap'] = [
        1 if len(np.intersect1d(x, seed_features['artists'])) else 0
        for x in candidate_df['artists']
    ]
    Xtrain['album_overlap'] = [
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
        Xtrain[f'{feat}_diff'] = diff

    Xtrain = pd.DataFrame.from_dict(Xtrain)
    return Xtrain, seed_features, candidate_df