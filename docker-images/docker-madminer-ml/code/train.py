#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import yaml
from pathlib import Path

from madminer.ml import ParameterizedRatioEstimator
from madminer.ml import ScoreEstimator


############################
##### Global variables #####
############################

project_dir = Path(__file__).parent.parent
models_dir = str(project_dir.joinpath('models'))


############################
##### Argument parsing #####
############################

samples_path = str(sys.argv[1])
input_file = str(sys.argv[2])

with open(input_file) as f:
    inputs = yaml.safe_load(f)


#############################
### Configuration parsing ###
#############################

path_split = os.path.split(os.path.abspath(samples_path))
sub_folder = path_split[1]
method = str(sub_folder.split("_", 3)[1])

alpha = float(inputs['alpha'])
batch_size = int(inputs['batch_size'])
num_epochs = int(inputs['n_epochs'])
valid_split = float(inputs['validation_split'])


############################
##### Perform training #####
############################

score_estimator_methods = {'sally', 'sallino'}
ratio_estimator_methods = {'alice', 'alices', 'cascal', 'carl', 'rolr', 'rascal'}

if method in score_estimator_methods:
    estimator = ScoreEstimator()
    estimator.train(
        method=method,
        x=samples_path + f'/x_{method}_train.npy',
        t_xz=samples_path + f'/t_xz_{method}_train.npy',
    )

elif method in ratio_estimator_methods:
    estimator = ParameterizedRatioEstimator(n_hidden=(100, 100, 100))
    estimator.train(
        method=method,
        alpha=alpha,
        theta=samples_path + f'/theta0_{method}_train.npy',
        x=samples_path + f'/x_{method}_train.npy',
        y=samples_path + f'/y_{method}_train.npy',
        r_xz=samples_path + f'/r_xz_{method}_train.npy',
        t_xz=samples_path + f'/t_xz_{method}_train.npy',
        n_epochs=num_epochs,
        validation_split=valid_split,
        batch_size=batch_size,
    )

else:
    raise ValueError('Invalid training method')


############################
#### Save trained model ####
############################

model_folder_name = method
model_folder_path = f'{models_dir}/{model_folder_name}'
os.makedirs(model_folder_path, exist_ok=True)

model_file_name = method
model_file_path = f'{model_folder_path}/{model_file_name}'
estimator.save(model_file_path)
