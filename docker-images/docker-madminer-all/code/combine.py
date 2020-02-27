from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
#import matplotlib
#from matplotlib import pyplot as plt
#%matplotlib inline
import sys 
import yaml
import six
import os
import logging
import h5py
import shutil


h5_list = sys.argv[1]

h5_list=h5_list.replace('[','')
h5_list=h5_list.replace(']','')
h5_list=h5_list.split()
h5_list=[str(file[1:]) for file in h5_list]

print("cleaning,...", h5_list)

FINAL_NAME = "combined_delphes.h5"
MERGE_ON = ["samples/observations","samples/weights"]

shutil.copy(h5_list[0], FINAL_NAME)
target_file = h5py.File(FINAL_NAME,'r+')

for merge_name in MERGE_ON:
	data_list = [np.array(h5py.File(name,'r')[merge_name]) for name in h5_list]
	merged_data = np.concatenate(data_list, axis=0)
	del target_file[merge_name]
	target_file.create_dataset(merge_name, data = merged_data)



# d_names = h5_list
# d_struct = {} #Here we will store the database structure
# for i in d_names:
#     print(i)
#     f = h5py.File(i,'r')
#     d_struct[i] = f.keys()
#     f.close()

# for i in d_names:
#     for j  in d_struct[i]:
#        os.system('h5copy -i %s -o combined_delphes.h5 -s %s -d %s' % (i, j, j))