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

# Constants
CLIENT_ID = '43da5b98ae53409e8584a09b01d0a83a' # Spotify API
CLIENT_SECRET = '3576a4f2a2604466b4c2f849e9a9df32' # Spotify API
INPUT_ALL_TRACKS_JSON = 'data/song_counts.json'
OUTPUT_ALL_TRACKS = 'data/spotify_api_database.pickle'
# Timing/Logging
LOG_BATCH_INTERVAL = 100 # Log every 100 batches
TIMER_START = None
def start_timer():
	global TIMER_START
	TIMER_START = time()
def lap_timer():
	global TIMER_START
	# Return time elapsed since TIMER_START was set, then reset TIMER_START
	elapsed = elapsed_time()
	TIMER_START = time()
	return elapsed
def elapsed_time(as_int = False):
	global TIMER_START
	# Return time elapsed since TIMER_START was set
	return "{0:.2f}s".format(time() - TIMER_START) if not as_int else time() - TIMER_START





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
			print(r.status_code, r.content)
			if r.status_code == 429:
				# Rate limiting
				seconds_to_wait = r.headers['Retry-After']
				print("Rate limited...Waiting "+str(seconds_to_wait)+" seconds")
				sleep(seconds_to_wait)
				# Retry request
				return self._get(endpoint, params)
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
		self.audio_feats = False # Set to TRUE when we update its audio features

	def add_artist(self, id, name):
		self.artists.append((id, name))

	def get_artists(self):
		return ', '.join([ str(a[1]) for a in self.artists])

	def get_audio_feats(self):
		if self.audio_feats:
			return "Danceability: " + str(self.danceability) + " | Energy: " + str(self.energy)
		return ""
	
	def __str__(self):
		return self.name + ', by ' + self.get_artists()

def fetchTracks(sp, track_ids, verbose = True):
	BATCH_SIZE = sp.lim_get_tracks # Fetch tracks 50 at a time (max allowed by Spotify API)
	N_BATCHES = int(np.ceil(len(track_ids)/BATCH_SIZE))
	start_time = time()
	tracks = []
	for batch_idx in range(0, N_BATCHES):
		batch_start_idx = batch_idx * BATCH_SIZE
		batch_end_idx = batch_start_idx + BATCH_SIZE
		batch = track_ids[batch_start_idx:batch_end_idx] if batch_end_idx < len(track_ids) else track_ids[batch_start_idx:]
		data = sp.get_tracks(batch)
		if data is None:
			# Premature failure - Return so we can save results so far
			if verbose:
				print("> Failed on track "+str(batch_start_idx))
			return tracks
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
				# If it fails, it's because "track" is None, so no point to logging it
				continue
		if verbose and batch_idx % LOG_BATCH_INTERVAL == 0:
			elapsed_time = time() - start_time
			start_time = time()
			remaining_time = (N_BATCHES - batch_idx - 1)/LOG_BATCH_INTERVAL * elapsed_time
			print("Fetched "+str(batch_end_idx) + "/" + str(len(track_ids))+ " tracks (" + "{0:.2f}s".format(elapsed_time) + " elapsed, " + "{0:.2f}s".format(remaining_time) + " left)")
	return tracks

def fetchAudioFeatures(sp, tracks, verbose = True):
	BATCH_SIZE = sp.lim_get_audio_feats # Fetch tracks 100 at a time (max allowed by Spotify API)
	N_BATCHES = int(np.ceil(len(tracks)/BATCH_SIZE))
	start_time = time()
	for batch_idx in range(0, N_BATCHES):
		batch_start_idx = batch_idx * BATCH_SIZE
		batch_end_idx = batch_start_idx + BATCH_SIZE
		batch = tracks[batch_start_idx:batch_end_idx] if batch_end_idx < len(tracks) else tracks[batch_start_idx:]
		data = sp.get_audio_features([ t.id for t in batch ])
		if data is None:
			# Premature failure - Return so we can save results so far
			if verbose:
				print("> Failed on track "+str(batch_start_idx))
		for idx, audio_feats in enumerate(data['audio_features']):
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
				track.audio_feats = True
			except Exception as e:
				print("  Failed fetchAudioFeatures() for: " + track.name + ", " + track.id)
				continue
		if verbose and batch_idx % LOG_BATCH_INTERVAL == 0:
			elapsed_time = time() - start_time
			start_time = time()
			remaining_time = (N_BATCHES - batch_idx - 1)/LOG_BATCH_INTERVAL * elapsed_time
			print("Fetched "+str(batch_end_idx) + "/" + str(len(tracks))+ " audio features (" + "{0:.2f}s".format(elapsed_time) + " elapsed, " + "{0:.2f}s".format(remaining_time) + " left)")

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

if __name__ == '__main__':
	# Set up SpotiPy object
	sp = SpotifyAuth()

	# Read in all tracks in MPD
	start_timer()
	with open(INPUT_ALL_TRACKS_JSON, 'r') as fd:
		all_tracks = json.load(fd)
	all_track_ids = sorted([ x[len('spotify:track:'):] for x in all_tracks.keys()]) # So that keys are always read in same order
	print("**** Read "+INPUT_ALL_TRACKS_JSON + ' in ' + lap_timer())

	# Read already saved Track() objs
	tracks = [] # All Track() objs in .pickle file
	complete_tracks = [] # Completed Track() with audio feats
	incomplete_tracks = [] # Track() with no audio feat
	complete_track_ids = []
	incomplete_track_ids = []
	with open(OUTPUT_ALL_TRACKS, 'rb') as fd:
		try:
			tracks = pickle.load(fd)
			for t in tracks:
				if t.audio_feats:
					complete_tracks.append(t)
					complete_track_ids.append(t.id)
				else:
					incomplete_tracks.append(t)
					incomplete_track_ids.append(t.id)
		except:
			print("Empty .pickle file")
	print("**** Read "+OUTPUT_ALL_TRACKS + ' in ' + lap_timer())
	partial_track_ids = complete_track_ids + incomplete_track_ids # Have Track() obj already saved
	need_track_ids = list(set(all_track_ids) - set(partial_track_ids))
	print(need_track_ids)
	# Fetch Track() objs for all MPD tracks that we haven't already pickled
	print("# of ... Completed Tracks: "+str(len(complete_tracks)) + " | Tracks Missing Audio: "+str(len(incomplete_tracks)) + " | Unfetched Tracks: "+str(len(need_track_ids)))
	need_audio_feats_tracks = fetchTracks(sp, need_track_ids)
	print(need_audio_feats_tracks)
	print("**** Fetched tracks in " + lap_timer())

	# Fetch audio feats for all Track() objs
	fetchAudioFeatures(sp, need_audio_feats_tracks + incomplete_tracks)
	print("**** Fetched audio features in " + lap_timer())
	tracks += need_audio_feats_tracks

	# Write objects to file
	with open(OUTPUT_ALL_TRACKS, 'wb') as fd:
		pickle.dump(tracks, fd)
	print("**** Wrote .pickle file in " + lap_timer())