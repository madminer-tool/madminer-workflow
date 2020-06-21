#!/usr/bin/python

import sys
from madminer import combine_and_shuffle
from pathlib import Path


##########################
#### Argument parsing ####
##########################

h5_list = sys.argv[1]
output_dir = Path(sys.argv[2])

file_path = output_dir.joinpath('data', 'combined_delphes.h5')


###########################
### Cleaning file names ###
###########################

h5_list = h5_list.replace('[', '')
h5_list = h5_list.replace(']', '')
h5_list = h5_list.split()
h5_list = [str(file) for file in h5_list]


##########################
#### Merging entities ####
##########################

combine_and_shuffle(
	input_filenames=h5_list,
	output_filename=file_path,
)
