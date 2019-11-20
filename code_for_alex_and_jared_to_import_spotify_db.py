import os
import sys
import pandas as pd
import numpy as np
import pickle
from time import time

from spotify_api_database import Track 

########################
########################
# NOTE: First, you need to go to: https://drive.google.com/open?id=14h1Hpdg1aLosORY6qRENhjTSqIt680SZ
#		and download the file named 'spotify_api_database.pickle'
#		Second, place this file in the "Data/" directory of this Github repo
########################
########################
INPUT_FILE = 'Data/spotify_api_database.pickle'


########################
########################
# Read in .pickle file
########################
########################

with open(INPUT_FILE, 'rb') as fd:
	start_time = time()
	print("Reading Spotify API Database .pickle file...")
	tracks = pickle.load(fd)
	print("Finished reading file (" + str(time() - start_time) +"s)...")

########################
########################
# The "tracks" variable is now an array of ~2M Track() objects
# The definition of the Track() class is in "spotify_api_database" -- note how we need to import this on line 7
########################
########################
# Print out first 100 tracks
for idx, t in enumerate(tracks):
	print(str(t) + ' | ' + t.get_audio_feats())
	# NOTE: Some tracks did not have audio feature data on Spotify (~600)
	if idx > 100:
		break
