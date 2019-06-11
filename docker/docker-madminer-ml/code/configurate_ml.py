from __future__ import absolute_import, division, print_function, unicode_literals

#import logging
import numpy as np
#import matplotlib
#from matplotlib import pyplot as plt
#%matplotlib inline

import yaml
import sys
import time
import logging
import multiprocessing
import h5py
from functools import partial

from madminer.core import MadMiner
from madminer.delphes import DelphesReader
from madminer.sampling import combine_and_shuffle
from madminer.sampling import SampleAugmenter
from madminer.sampling import benchmark, benchmarks, random_morphing_points, morphing_point
from madminer.ml import ParameterizedRatioEstimator, DoubleParameterizedRatioEstimator, LikelihoodEstimator, ScoreEstimator
from madminer.plotting import plot_2d_morphing_basis, plot_distributions

from madminer.analysis import DataAnalyzer
from madminer.utils.interfaces.madminer_hdf5 import madminer_event_loader
from madminer.utils.interfaces.madminer_hdf5 import save_preformatted_events_to_madminer_file
from madminer.utils.various import create_missing_folders, shuffle

logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.INFO
)

for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)


n_trainsamples = int(sys.argv[1])

h5_file = sys.argv[2]

inputs_file = sys.argv[3]
with open(inputs_file) as f:
    inputs = yaml.safe_load(f)


nuisance = inputs['include_nuisance_parameters']

methods = inputs['methods']
print(methods)
methods = map(lambda x: str(x), methods)

test_split=float(inputs['test_split']) #training-test split

# get number of paramenters
hf = h5py.File(h5_file, 'r')
parameters = len(hf['parameters']['names'])


#to shuffle or not to shuffle
if (inputs['shuffle']): 
    h5shuffle_file = '/home/data/madminer_example_shuffled.h5'
    
    combine_and_shuffle(
        [h5_file],
        h5shuffle_file 
    )

    sampler = SampleAugmenter(h5shuffle_file, include_nuisance_parameters=nuisance)  #'data/madminer_example_shuffled.h5'

else:
    sampler = SampleAugmenter(h5_file, include_nuisance_parameters=nuisance)



for method in methods:
    print('sampling from method ', method)

    for i in range(n_trainsamples):

        # creates training samples

        # different methods have different arguments
        # TRAIN RATIO
        
        if method in ['alice','alices','cascal','carl','rolr', 'rascal']:
            theta0_sampling = inputs[str(method)]['theta_0']['sampling_method'] #sampling method for theta0
            theta1_sampling = inputs[str(method)]['theta_1']['sampling_method'] #sampling method for theta1
            theta_0 = inputs[str(method)]['theta_0'] #parameters for theta0 sampling
            theta_1 = inputs[str(method)]['theta_1'] #parameters for theta0 sampling

             ##random_morphing_points has two arguments not one
            if (theta0_sampling == 'random_morphing_points' and theta1_sampling != 'random_morphing_points' ): 
                
                prior = []

                for p in range(parameters):
                    this_tuple = theta_0['prior']['parameter_'+str(p)]
                    prior.append( (str(this_tuple['prior_shape']), float(this_tuple['prior_param_0']), float(this_tuple['prior_param_1'])) )


                _ = sampler.sample_train_ratio(
                    theta0=eval(theta0_sampling)(theta_0['n_thetas'], prior),
                    theta1=eval(theta1_sampling)(theta_1['argument']),
                    n_samples=int(inputs['n_samples']['train']),
                    folder='/home/data/Samples_'+str(method)+'_'+str(i),
                    filename=method+'_train'      
                )

            elif (theta1_sampling == 'random_morphing_points' and theta0_sampling != 'random_morphing_points'):  
                tuple_0 = theta_1['prior']['parameter_0'] #tuple for parameter 0
                tuple_1 = theta_1['prior']['parameter_1'] #tuple for parameter 1
                prior = [ (str(tuple_0['prior_shape']), float(tuple_0['prior_param_0']), float(tuple_0['prior_param_1'])), \
                           (str(tuple_1['prior_shape']), float(tuple_1['prior_param_0']), float(tuple_1['prior_param_1']))  ]

                x, theta0, theta1, y, r_xz, t_xz = sampler.sample_train_ratio(
                    theta0=eval(theta0_sampling)(theta_0['argument']),
                    theta1=eval(theta1_sampling)(theta_1['n_thetas'], prior),
                    n_samples=int(inputs['n_samples']['train']),
                    folder='/home/data/Samples_'+str(method)+'_'+str(i),
                    filename=method+'_train'      
                )

            elif (theta0_sampling == 'random_morphing_points' and theta1_sampling == 'random_morphing_points'): 
                tuple0_0 = theta_0['prior']['parameter_0'] #tuple for parameter 0
                tuple0_1 = theta_0['prior']['parameter_1'] #tuple for parameter 1
                prior0 = [ (str(tuple0_0['prior_shape']), float(tuple0_0['prior_param_0']), float(tuple0_0['prior_param_1'])), \
                           (str(tuple0_1['prior_shape']), float(tuple0_1['prior_param_0']), float(tuple0_1['prior_param_1']))  ]
                
                tuple1_0 = theta_1[method]['prior']['parameter_0'] #tuple for parameter 0
                tuple1_1 = theta_1[method]['prior']['parameter_1'] #tuple for parameter 1
                prior1 = [ (str(tuple1_0['prior_shape']), float(tuple1_0['prior_param_0']), float(tuple1_0['prior_param_1'])), \
                           (str(tuple1_1['prior_shape']), float(tuple1_1['prior_param_0']), float(tuple1_1['prior_param_1']))  ]

                x, theta0, theta1, y, r_xz, t_xz = sampler.sample_train_ratio(
                    theta0=eval(theta0_sampling)(theta_0['n_thetas'], prior0),
                    theta1=eval(theta1_sampling)(theta_1['n_thetas'], prior1),
                    n_samples=int(inputs['n_samples']['train']),
                    folder='/home/data/Samples_'+str(method)+'_'+str(i),
                    filename=method+'_train'      
                )

            else:
                x, theta0, theta1, y, r_xz, t_xz = sampler.sample_train_ratio(
                    theta0=benchmark('w'),
                    theta1=benchmark('sm'),
                    n_samples=int(inputs['n_samples']['train']),
                    folder='/home/data/Samples_'+str(method)+'_'+str(i),
                    filename=method+'_train'
                )
                 #x, theta0, theta1, y, r_xz, t_xz = sampler.sample_train_ratio(
    #             #    theta0=eval(theta0_sampling)(theta_0['argument']),
    #             #    theta1=eval(theta1_sampling)(theta_1['argument']),
    #             #    n_samples=int(inputs['n_samples']['train']),
    #             #    test_split=test_split,
    #             #    folder='/home/data/Samples_'+str(method)+'_'+str(i),
    #             #    filename=method+'_train'
    #             #)


    #     #TRAIN LOCAL
        if method in ['sally','sallino']:
            theta_input = inputs[str(method)]['theta']
            theta_sampling = theta_input['sampling_method']
            #parameters for theta  sampling

            if (theta_sampling == 'random_morphing_points'): 
                
                prior = []
                for p in range(parameters):
                    this_tuple = theta_input['prior']['parameter_'+str(p)]
                    prior.append( (str(this_tuple['prior_shape']), float(this_tuple['prior_param_0']), float(this_tuple['prior_param_1'])) )


                x, theta0, theta1, y, r_xz, t_xz = sample_train_local(
                    theta=eval(theta_sampling)(theta_input['n_thetas'], prior),
                    n_samples=int(inputs['n_samples']['train']),
                    folder='/home/data/Samples_'+str(method)+'_'+str(i),
                    filename=method+'_train'      
                )

            if (theta_sampling == 'benchmark'): 
                _ = sampler.sample_train_local(
                    theta=eval(theta_sampling)(theta_input['argument']),
                    n_samples=int(inputs['n_samples']['train']),
                    folder='/home/data/Samples_'+str(method)+'_'+str(i),
                    filename=method+'_train'
                )

    #     #TRAIN GLOBAL
        if method in ['scandal']:
            theta_sampling = inputs['theta']['sampling_method']
            theta = inputs[str(method)]['theta'] #parameters for theta sampling

            if ( theta_sampling == 'random_morphing_points' ): 
                tuple_0 = theta_sampling['prior']['parameter_0'] #tuple for parameter 0
                tuple_1 = theta_sampling['prior']['parameter_1'] #tuple for parameter 1
                prior = [ (str(tuple_0['prior_shape']), float(tuple_0['prior_param_0']), float(tuple_0['prior_param_1'])), \
                           (str(tuple_1['prior_shape']), float(tuple_1['prior_param_0']), float(tuple_1['prior_param_1']))  ] 
                
                x, theta0, theta1, y, r_xz, t_xz = train_samples_density(
                    theta=eval(theta_sampling)(theta['n_thetas'], prior),
                    n_samples=int(inputs['n_samples']['train']),
                    folder='/home/data/Samples_'+str(method)+'_'+str(i),
                    filename=method+'_train'      
                )

            else:
                x, theta0, theta1, y, r_xz, t_xz = sampler.train_samples_density(
                    theta=eval(theta_sampling)(theta['argument']),
                    n_samples=int(inputs['n_samples']['train']),
                    folder='/home/data/Samples_'+str(method)+'_'+str(i),
                    filename=method+'_train'
                )
    


