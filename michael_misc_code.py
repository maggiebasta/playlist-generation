
class Album():
	def __init__(self, id, name, trackCount):
		self.id = id
		self.name = name
		self.trackCount = trackCount
		self.tracks = [] # Array of Track objects

	def addTrack(self, track):
		self.tracks.append(track)

class Artist():
	def __init__(self, id, name):
		self.id = id
		self.name = name
		self.albums = [] # Array of Album objects

	def __str__(self):
		return self.name

class Playlist():
	def __init__(self, id, name, trackCount):
		self.id = id
		self.name = name
		self.trackCount = trackCount
		self.tracks = [] # Array of Track objects

	def addTrack(self, track):
		self.tracks.append(track)

	def __str__(self):
		return 'PLAYLIST: '+self.name + ' | TRACKS: ' + ', '.join([ str(t) for t in self.tracks ])

def fetchPlaylistTracks(username, playlist):
	data = sp.user_playlist_tracks(username, playlist_id=playlist.id, fields=None, limit=100)
	while data:
		for idx, track in enumerate(data['items']):
			id = ''
			name = ''
			artists = []
			album = ''
			albumCoverURL = ''
			try:
				id = track['track']['id']
				name = track['track']['name']
				artists = [ Artist(x['id'], x['name']) for x in track['track']['artists'] ]
				album = track['track']['album']['name']
				albumCoverURL = track['track']['album']['images'][0]['url'] # Get most hi-def image
			except:
				pass
			if id == '' or name == '' or len(artists) < 1:
				print("Error with track "+id +" "+name + ' ' + ''.join(artists))
			else:
				tObj = Track(id, name, album, albumCoverURL)
				for a in artists:
					tObj.addArtist(a)
				playlist.addTrack(tObj)
		if data['next']:
			data = sp.next(data)
		else:
			data = None
	return True

def fetchFeaturedPlaylists():
	results = []
	data = sp.featured_playlists(limit = 50)['playlists']
	while data:
		for idx, playlist in enumerate(data['items']):
			id = playlist['id']
			name = playlist['name']
			trackCount = playlist['tracks']['total']
			pObj = Playlist(id, name, trackCount)
			results.append(pObj)
		if data['next']:
			data = sp.next(data)
		else:
			data = None
	return results

def fetchArtistAlbums(artist):
	results = []
	data = sp.artist_albums(limit = 50)
	while data:
		for idx, playlist in enumerate(data['items']):
			id = playlist['id']
			name = playlist['name']
			playlist['release_date']
			pObj = Playlist(id, name, trackCount)
			results.append(pObj)
		if data['next']:
			data = sp.next(data)
		else:
			data = None
	return results

def fetchRelatedArtists(artist):
	results = []
	data = sp.artist_related_artists(artist.id)
	for idx, related_artist in enumerate(data):
		id = related_artist['id']
		name = related_artist['name']
		new_artist_obj = Artist(id, name)
		new_artist_obj.popularity = related_artist['popularity']
		new_artist_obj.genres = related_artist['genres']
		new_artist_obj.n_followers = related_artist['n_followers']['total']
		results.append(new_artist_obj)
	return results