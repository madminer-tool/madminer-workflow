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
from collections import OrderedDict

from madminer.core import MadMiner
from madminer.delphes import DelphesReader
from madminer.sampling import combine_and_shuffle

from madminer.utils.interfaces.madminer_hdf5 import (
    save_events_to_madminer_file,
    load_madminer_settings,
    save_nuisance_setup_to_madminer_file,
)
from madminer.utils.interfaces.delphes import run_delphes
from madminer.utils.interfaces.delphes_root import parse_delphes_root_file
from madminer.utils.interfaces.hepmc import extract_weight_order
from madminer.utils.interfaces.lhe import parse_lhe_file, extract_nuisance_parameters_from_lhe_file

logger = logging.getLogger(__name__)


h5_file = sys.argv[1]
event_path = sys.argv[2]
input_file = sys.argv[3]
benchmark_file = sys.argv[4]

mg_dir = '/home/software/MG5_aMC_v2_6_2'

dp = DelphesReader(h5_file)

#### get benchmark name of the job
file = open(benchmark_file,'r')
benchmark=str(file.read())
print(benchmark)
print(type(benchmark))
####

dp.add_sample(
    lhe_filename=event_path + '/unweighted_events.lhe.gz', #'/home/code/mg_processes/signal/Events/run_01/unweighted_events.lhe.gz'
    hepmc_filename=event_path + '/tag_1_pythia8_events.hepmc.gz', #'/home/code/mg_processes/signal/Events/run_01/tag_1_pythia8_events.hepmc.gz'
    is_background=False,
    sampled_from_benchmark=benchmark,#'sm',
    weights='lhe'
)

dp.run_delphes(
    delphes_directory=mg_dir + '/Delphes',
    delphes_card='/home/code/cards/delphes_card.dat',
    log_file='/home/log_delphes.log')
    #initial_command='source activate python2'



########### add observables and cuts from input file
with open(input_file) as f:
    # use safe_load instead load
    dict_all = yaml.safe_load(f)

for observable in dict_all['observables']:
	dp.add_observable(
    observable['name'],
    observable['definition'],
    required=observable['required'],
    default=observable['default']
	)

for cut in dict_all['cuts']:
    dp.add_cut(cut['expression'])
#####################################

dp.analyse_delphes_samples()#reference_benchmark=benchmark

dp.save("/home/data/madminer_example_with_data_"+str(benchmark)+".h5")
