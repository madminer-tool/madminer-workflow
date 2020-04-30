#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

from madminer import combine_and_shuffle
from pathlib import Path


##########################
#### Global variables ####
##########################

project_dir = Path(__file__).parent.parent

data_dir = project_dir.joinpath('data')
file_path = data_dir.joinpath("combined_delphes.h5")


##########################
#### Argument parsing ####
##########################

h5_list = sys.argv[1]


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
