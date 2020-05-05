#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import yaml
from madminer import DelphesReader
from pathlib import Path


##########################
#### Global variables ####
##########################

project_dir = Path(__file__).parent.parent

data_dir = str(project_dir.joinpath('data'))
card_dir = str(project_dir.joinpath('code', 'cards'))
logs_dir = str(project_dir.joinpath('code', 'logs'))
madg_dir = str(project_dir.joinpath('software', 'MG5_aMC_v2_6_7'))


########################
### Argument parsing ###
########################

config_file = sys.argv[1]
event_path = sys.argv[2]
input_file = sys.argv[3]
benchmark_file = sys.argv[4]

with open(input_file, 'r') as f:
    spec = yaml.safe_load(f)

with open(benchmark_file, 'r') as f:
    benchmark = f.read()


########################
## Load configuration ##
########################

reader = DelphesReader(config_file)


#########################
###### Run Delphes ######
#########################

reader.add_sample(
    lhe_filename=event_path + '/unweighted_events.lhe.gz',
    hepmc_filename=event_path + '/tag_1_pythia8_events.hepmc.gz',
    sampled_from_benchmark=benchmark,
    weights='lhe',
)

reader.run_delphes(
    delphes_directory=madg_dir + '/' + 'Delphes',
    delphes_card=card_dir + '/' + 'delphes_card.dat',
    log_file=logs_dir + '/' + 'log_delphes.log',
)


############################
## Add observables / cuts ##
############################

for observable in spec['observables']:
    reader.add_observable(
        observable['name'],
        observable['definition'],
        observable['required'],
        observable['default'],
    )

for cut in spec['cuts']:
    reader.add_cut(cut['expression'])


############################
###### Analyse events ######
############################

reader.analyse_delphes_samples()


############################
##### Save events data #####
############################

os.makedirs(data_dir, exist_ok=True)

data_file_name = f'madminer_delphes_data_{benchmark}.h5'
data_file_path = data_dir + '/' + data_file_name

reader.save(data_file_path)
