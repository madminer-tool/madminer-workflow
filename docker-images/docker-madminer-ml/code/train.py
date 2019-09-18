from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import yaml
import sys
import os
import logging
import math

from madminer.core import MadMiner
from madminer import sampling
from madminer.sampling import SampleAugmenter
from madminer.ml import ParameterizedRatioEstimator, ScoreEstimator, Ensemble
from madminer.plotting import plot_2d_morphing_basis, plot_distributions

samples_path = str(sys.argv[1])

input_file = sys.argv[2]
with open(input_file) as f:
    inputs = yaml.safe_load(f)


# get method from inputs
path_split = os.path.split(os.path.abspath(samples_path))

sub_folder = path_split[1] 

method = str(sub_folder.split("_", 3)[1])


# training options

if(method in ['sally', 'sallino']):
    estimator = ScoreEstimator()
    estimator.train(
        method=method,
        x=samples_path+'/x_'+method+'_train.npy',
        t_xz=samples_path+'/t_xz_'+method+'_train.npy',
    )
    os.mkdir('/madminer/models/'+method)
    estimator.save('/madminer/models/'+method+'/'+method)


if(method in ['alice','alices','cascal','carl','rolr', 'rascal']):
    estimator = ParameterizedRatioEstimator(n_hidden=(100,100,100))
    estimator.train(
    method=method,
    alpha=float(inputs['alpha']),
    theta=samples_path+'/theta0_'+method+'_train.npy',
    x=samples_path+'/x_'+method+'_train.npy',
    y=samples_path+'/y_'+method+'_train.npy',
    r_xz=samples_path+'/r_xz_'+method+'_train.npy',
    t_xz=samples_path+'/t_xz_'+method+'_train.npy',
    n_epochs=int(inputs['n_epochs']),
    validation_split=float(inputs['validation_split']),
    batch_size=int(inputs['batch_size'])
    )
    os.mkdir('/madminer/models/'+method)
    estimator.save('/madminer/models/'+method+'/'+method)

