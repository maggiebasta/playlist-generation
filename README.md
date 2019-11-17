# CS 109 Final Project: Spotify Playlist Generation

## Data Decription 
- `spotify_api_database.pickle`: Pickle file of Track() objects with Spotify API annotations -- too large for Github (>800MB), but accessible [at this link in the Drive at Data/spotify_api_database.pickle](https://drive.google.com/open?id=14h1Hpdg1aLosORY6qRENhjTSqIt680SZ). The 2262292 tracks in the MPD were broken down in the following numbers in `spotify_api_database.pickle`:
	- 2261603: Have complete Spotify API Track (e.g. artist, name) and Audio features (e.g. danceability, loudness)
	- 106: Have Spotify API Track but no Audio features
	- 588: Were not accessible via Spotify API, and thus are not present in `spotify_api_database.pickle`
- `song_counts.json`: JSON with key=track UID, value=appearences in MPD playlists 
- `top_songs.npy`: array of songs that appear in > 10 playlists (about 10% of 2,262,292 total)
- `song_uid2name.json.zip`: compression JSON for lookup of song names. Key=UID, value=song name. (The full json, which is output by `top_songs_extract.ipynb` is too large to store in github, so it is put in the `.gitignore` file and the extracted version is uploaded instead). 

*NOTE: in order to run the notebooks, the data directory should also include the MDP csv's in "data/Songs/" (downloaded from google drive).*

## Notebooks
- `initial_EDA.ipynb`: the very first inspection of the MPD (used for milestone 2)
- `song_distance_calculator.ipynb`: calculates the "distance" between songs (i.e. inverse of the frequency they appear in playlists together) 
- `top_songs_extract.ipynb`: extracts the "top" songs that appear in the MPD and stores them in `data/top_songs.npy` (i.e. songs that appear in > 10 playlists, about 10% of 2,262,292 total). The notebook also stores the counts of song occurences in `data/song_counts.json` and UID to song name lookup in `data/song_uid2name.json`. 

## Data Generation/Exploration Workflow
To keep track of the work we've done and make our eventual write-up easier.

1. Run `initial_EDA.ipynb` for data exploration of MPD
2. Run `top_songs_extract.ipynb` to get a dictionary of counts for each track ID (stored in `data/song_counts.json`)
3. Run `gen_spotify_api_database.py` to fetch audio features and additional track information for each track via the Spotify API. Results stored as **Track()** objects in `spotify_api_database.pickle` on the Google Drive
4. Run `song_distance_calculator.ipynb` to calculate "distance" between songs -- **???? <<Michael question for @Maggie>> is this just a 2M x 2M matrix where matrix[i][j] = # of times song i and song j appear in same playlist ???? (stored in ????)**


## Models

### WRMF

Stored in `Models/wrmf.py`

- Use Weighted Regularized Matrix Factorization to get most similar tracks based on their presence in playlists
	- User x Item matrix ==> Playlist x Track matrix (where M[i][j] = 1 iff Playlist i contains Track j)
- Currently uses dot product to assess similarity between tracks -- Future work: However, this paper advises to use cosine similarity instead of simple dot product (http://www.cs.toronto.edu/~mvolkovs/sigir2015_svd.pdf)

### Nearest Neighbor

Stored in `Models/nearest_neighbor.py`

- Use K-NN with distance metric of cosine similarity on a Track x Playlist matrix (where M[i][j] = 1 iff Track i is in Playlist j) to get similar tracks
