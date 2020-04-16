#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import sys
import h5py
import shutil


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
h5_list = [str(file[1:]) for file in h5_list]

print(f"Cleaning... {h5_list}")


##########################
#### Merging entities ####
##########################

FINAL_NAME = "combined_delphes.h5"
MERGE_ON = ["samples/observations", "samples/weights"]

shutil.copy(h5_list[0], FINAL_NAME)
target_file = h5py.File(FINAL_NAME, 'r+')

for merge_name in MERGE_ON:
	data_list = [np.array(h5py.File(name, 'r')[merge_name]) for name in h5_list]
	merged_data = np.concatenate(data_list, axis=0)

	del target_file[merge_name]
	target_file.create_dataset(merge_name, data=merged_data)
