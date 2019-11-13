# playlist-generation

## Data Decription 
- song_counts.json: JSON with key=track UID, value=appearences in MPD playlists 
- top_songs.npy: array of songs that appear in > 10 playlists (about 10% of 2,262,292 total)
- song_uid2name.json.zip: compression JSON for lookup of song names. Key=UID, value=song name
*NOTE: in order to run the notebooks, the data directory should also include the MDP csv's in "data/Songs/" (downloaded from google drive). 

## Notebooks
- initial_EDA.ipynb: the very first inspection of the MPD (used for milestone 2)
- song_distance_calculator.ipynb: calculates the "distance" between songs (i.e. inverse of the frequency they appear in playlists together) 
- top_songs_extract.ipynb: extracts the "top" songs that appear in the MPD and stores them in data/top_songs.npy (i.e. songs that appear in > 10 playlists, about 10% of 2,262,292 total). The notebook also stores the counts of song occurences in data/song_counts.json and UID to song name lookup in data/song_uid2name.json. 
