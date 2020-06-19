#!/usr/bin/python

import os
import sys
import yaml
from ast import literal_eval
from madminer import MadMiner
from pathlib import Path


##########################
#### Argument parsing ####
##########################

input_file = Path(sys.argv[1])
output_dir = Path(sys.argv[2])

data_dir = str(output_dir.joinpath('data'))

with open(input_file, 'r') as f:
    spec = yaml.safe_load(f)


###########################
### Miner configuration ###
###########################

miner = MadMiner()

# Add parameters
for parameter in spec['parameters']:
    param_range = parameter.pop('parameter_range')
    param_range = literal_eval(param_range)
    param_range = [float(val) for val in param_range]

    miner.add_parameter(**parameter, parameter_range=tuple(param_range))

# Add benchmarks
for benchmark in spec['benchmarks']:
    param_values = {}

    for i, _ in enumerate(spec['parameters']):
        name = benchmark[f'parameter_name_{i+1}']
        value = benchmark[f'value_{i+1}']
        param_values[name] = value

    miner.add_benchmark(param_values, benchmark['name'])


##########################
#### Morphing setting ####
##########################

miner.set_morphing(**spec['set_morphing'])


##########################
### Save configuration ###
##########################

os.makedirs(data_dir, exist_ok=True)

config_file_name = 'madminer_config.h5'
config_file_path = data_dir + '/' + config_file_name

miner.save(config_file_path)
