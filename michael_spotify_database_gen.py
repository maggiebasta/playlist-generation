import sys
import requests
import bs4
import traceback
import numpy as np
import json
import base64
from time import time
import threading
import pickle

CLIENT_ID = '43da5b98ae53409e8584a09b01d0a83a' # Spotify API
CLIENT_SECRET = '3576a4f2a2604466b4c2f849e9a9df32' # Spotify API
INPUT_ALL_TRACKS_JSON = 'data/song_counts.json'
OUTPUT_ALL_TRACKS = 'data/michael_song_audio.pickle'

class Spotify():
	def __init__(self, access_token, expires):
		self.access_token = access_token
		self.expires = expires
		self.lim_get_tracks = 50
		self.lim_get_audio_feats = 100

	def _get(self, endpoint, params):
		headers = { 'Authorization' : 'Bearer ' + self.access_token }
		r = requests.get(endpoint, headers = headers, params = params)
		if r.status_code != 200:
			if r.status_code == 429:
				# Rate limiting
				seconds_to_wait = r.headers['Retry-After']
				print("Rate limited...Waiting "+str(seconds_to_wait)+" seconds")
				sleep(seconds_to_wait)
				# Retry request
				return self._get(endpoint, params)
			print(r.content)
			print(r.status_code)
			return None
		return r.json()

	def get_tracks(self, track_ids):
		# Max 50 track_ids
		params = { 'ids' : ','.join(track_ids) }
		return self._get('https://api.spotify.com/v1/tracks', params)

	def get_audio_features(self, track_ids):
		# Max 100 track_ids
		params = { 'ids' : ','.join(track_ids) }
		return self._get('https://api.spotify.com/v1/audio-features', params)

class Track():
	def __init__(self, id, name):
		self.id = id
		self.name = name
		self.artists = [] # Array of Artist objects
		self.album = ''

	def add_artist(self, id, name):
		self.artists.append((id, name))

	def __str__(self):
		return self.name + ', by '+', '.join([ str(a[1]) for a in self.artists]) + ' ('+self.album+')'

def fetchTracks(sp, track_ids):
	tracks = []
	BATCH_SIZE = sp.lim_get_tracks # Fetch tracks 50 at a time (max allowed by Spotify API)
	for batch_idx in range(0, int(np.ceil(len(track_ids)/BATCH_SIZE))):
		batch_start_idx = batch_idx * BATCH_SIZE
		batch_end_idx = batch_start_idx + BATCH_SIZE
		batch = track_ids[batch_start_idx:batch_end_idx] if batch_end_idx < len(track_ids) else track_ids[batch_start_idx:]
		data = sp.get_tracks(batch)
		for idx, track in enumerate(data['tracks']):
			try:
				obj = Track(track['id'], track['name'])
				obj.popularity = track['popularity']
				obj.explicit = track['explicit']
				# Album
				if 'album' in track:
					obj.album_id = track['album']['id']
					obj.album_name = track['album']['name']
				# Artists
				if 'artists' in track:
					for artist in track['artists']:
						obj.add_artist(artist['id'], artist['name'])
				# Save Track Obj
				tracks.append(obj)
			except Exception as e:
				continue
		print("Fetched "+str(batch_end_idx) + "/" + str(len(track_ids))+ " tracks")
	return tracks

def fetchAudioFeatures(sp, tracks):
	BATCH_SIZE = sp.lim_get_audio_feats # Fetch tracks 100 at a time (max allowed by Spotify API)
	for batch_idx in range(0, int(np.ceil(len(tracks)/BATCH_SIZE))):
		batch_start_idx = batch_idx * BATCH_SIZE
		batch_end_idx = batch_start_idx + BATCH_SIZE
		batch = tracks[batch_start_idx:batch_end_idx] if batch_end_idx < len(tracks) else tracks[batch_start_idx:]
		data = sp.get_audio_features([ t.id for t in batch ])['audio_features']
		for idx, audio_feats in enumerate(data):
			try:
				track = tracks[batch_start_idx + idx]
				track.danceability = audio_feats['danceability']
				track.energy = audio_feats['energy']
				track.key = audio_feats['key']
				track.loudness = audio_feats['loudness']
				track.mode = audio_feats['mode']
				track.speechiness = audio_feats['speechiness']
				track.acousticness = audio_feats['acousticness']
				track.instrumentalness = audio_feats['instrumentalness']
				track.liveness = audio_feats['liveness']
				track.valence = audio_feats['valence']
				track.tempo = audio_feats['tempo']
				track.duration_ms = audio_feats['duration_ms']
				track.time_signature = audio_feats['time_signature']
			except Exception as e:
				print("  Failed for: " + track.name + ", " + track.id)
				continue
		print("Fetched "+str(batch_end_idx) + "/" + str(len(tracks))+ " audio features")

def SpotifyAuth():
	auth_string = CLIENT_ID + ':' + CLIENT_SECRET
	headers = { 'Authorization' : 'Basic '+ base64.b64encode(auth_string.encode('ascii')).decode('utf-8') }
	data = { 'grant_type' : 'client_credentials' }
	r = requests.post('https://accounts.spotify.com/api/token', headers = headers, data = data)
	if r.status_code != 200:
		return False
	json = r.json()
	return Spotify(json['access_token'], time() + json['expires_in'])

####################
## ACTUAL PROGRAM ##
####################

# Set up SpotiPy object
sp = SpotifyAuth()

# Read in all tracks in MPD
with open(INPUT_ALL_TRACKS_JSON, 'r') as fd:
	all_tracks = json.loads(fd.read())
all_track_ids = sorted([ x[len('spotify:track:'):] for x in all_tracks.keys()]) # So that keys are always read in same order
print("Read "+INPUT_ALL_TRACKS_JSON)

# Fetch Track() objs for all MPD tracks
tracks = fetchTracks(sp, all_track_ids[0:300])

# Fetch audio feats for all Track() objs
fetchAudioFeatures(sp, tracks)

# Write objects to file
with open(OUTPUT_ALL_TRACKS, 'wb') as fd:
	pickle.dump(tracks, fd)


# Read objects from file
with open(OUTPUT_ALL_TRACKS, 'rb') as fd:
	tracks = pickle.load(fd)