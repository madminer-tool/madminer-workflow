from __future__ import absolute_import, division, print_function, unicode_literals

#import logging
import numpy as np
#import matplotlib
#from matplotlib import pyplot as plt
#%matplotlib inline

import yaml
import sys

from madminer.core import MadMiner
from madminer.delphes import DelphesProcessor
from madminer.sampling import combine_and_shuffle
from madminer.sampling import SampleAugmenter
from madminer.sampling import constant_benchmark_theta, multiple_benchmark_thetas, random_morphing_thetas
from madminer.ml import MLForge
from madminer.plotting import plot_2d_morphing_basis, plot_distributions

n_trainsamples = int(sys.argv[1])

h5_file = sys.argv[2]

inputs_file = sys.argv[3]
with open(inputs_file) as f:
    inputs = yaml.safe_load(f)

#to shuffle or not to shuffle
if (inputs['shuffle']): 
    h5shuffle_file = '/home/data/madminer_example_shuffled.h5'
    
    combine_and_shuffle(
        [h5_file],
        h5shuffle_file 
    )

    sa = SampleAugmenter(h5shuffle_file)  #'data/madminer_example_shuffled.h5'

else:
    sa = SampleAugmenter(h5_file)


method=inputs['method']

test_split=float(inputs['test_split']) #training-test split

for i in range(n_trainsamples):
    #creates training samples

    #different methods have different arguments
    #TRAIN RATIO
    if method in ['alice','alices','cascal','carl','rolr', 'rascal']:
        theta0_sampling = inputs['theta_0']['sampling_method'] #sampling method for theta0
        theta1_sampling = inputs['theta_1']['sampling_method'] #sampling method for theta1
        theta_0 = inputs['theta_0'] #parameters for theta0 sampling
        theta_1 = inputs['theta_1'] #parameters for theta0 sampling

        ##random_morphing_thetas has two arguments not one
        if (theta0_sampling == 'random_morphing_thetas' and theta1_sampling != 'random_morphing_thetas' ): 
            tuple_0 = theta_0['prior']['parameter_0'] #tuple for parameter 0
            tuple_1 = theta_0['prior']['parameter_1'] #tuple for parameter 1
            prior = [ (str(tuple_0['prior_shape']), float(tuple_0['prior_param_0']), float(tuple_0['prior_param_1'])), \
                      (str(tuple_1['prior_shape']), float(tuple_1['prior_param_0']), float(tuple_1['prior_param_1']))  ] 
            
            print('tests debugging')
            print( eval(theta0_sampling)(theta_0['n_thetas'], prior) )


            x, theta0, theta1, y, r_xz, t_xz = sa.extract_samples_train_ratio(
                theta0=eval(theta0_sampling)(theta_0['n_thetas'], prior),
                theta1=eval(theta1_sampling)(theta_1['argument']),
                n_samples=int(inputs['n_samples']['train']),
                test_split=test_split,
                folder='/home/data/Samples_'+str(i),
                filename='train'      
            )

        elif (theta1_sampling == 'random_morphing_thetas' and theta0_sampling != 'random_morphing_thetas'):  
            tuple_0 = theta_1['prior']['parameter_0'] #tuple for parameter 0
            tuple_1 = theta_1['prior']['parameter_1'] #tuple for parameter 1
            prior = [ (str(tuple_0['prior_shape']), float(tuple_0['prior_param_0']), float(tuple_0['prior_param_1'])), \
                      (str(tuple_1['prior_shape']), float(tuple_1['prior_param_0']), float(tuple_1['prior_param_1']))  ]

            x, theta0, theta1, y, r_xz, t_xz = sa.extract_samples_train_ratio(
                theta0=eval(theta0_sampling)(theta_0['argument']),
                theta1=eval(theta1_sampling)(theta_1['n_thetas'], prior),
                n_samples=int(inputs['n_samples']['train']),
                test_split=test_split,
                folder='/home/data/Samples_'+str(i),
                filename='train'      
            )

        elif (theta0_sampling == 'random_morphing_thetas' and theta1_sampling == 'random_morphing_thetas'): 
            tuple0_0 = theta_0['prior']['parameter_0'] #tuple for parameter 0
            tuple0_1 = theta_0['prior']['parameter_1'] #tuple for parameter 1
            prior0 = [ (str(tuple0_0['prior_shape']), float(tuple0_0['prior_param_0']), float(tuple0_0['prior_param_1'])), \
                      (str(tuple0_1['prior_shape']), float(tuple0_1['prior_param_0']), float(tuple0_1['prior_param_1']))  ]
            
            tuple1_0 = theta_1['prior']['parameter_0'] #tuple for parameter 0
            tuple1_1 = theta_1['prior']['parameter_1'] #tuple for parameter 1
            prior1 = [ (str(tuple1_0['prior_shape']), float(tuple1_0['prior_param_0']), float(tuple1_0['prior_param_1'])), \
                      (str(tuple1_1['prior_shape']), float(tuple1_1['prior_param_0']), float(tuple1_1['prior_param_1']))  ]

            x, theta0, theta1, y, r_xz, t_xz = sa.extract_samples_train_ratio(
                theta0=eval(theta0_sampling)(theta_0['n_thetas'], prior0),
                theta1=eval(theta1_sampling)(theta_1['n_thetas'], prior1),
                n_samples=int(inputs['n_samples']['train']),
                test_split=test_split,
                folder='/home/data/Samples_'+str(i),
                filename='train'      
            )

        else:
            x, theta0, theta1, y, r_xz, t_xz = sa.extract_samples_train_ratio(
                theta0=eval(theta0_sampling)(theta_0['argument']),
                theta1=eval(theta1_sampling)(theta_1['argument']),
                n_samples=int(inputs['n_samples']['train']),
                test_split=test_split,
                folder='/home/data/Samples_'+str(i),
                filename='train'      
            )


    #TRAIN LOCAL
    if method in ['sally','sallino']:
        theta_sampling = inputs['theta']['sampling_method']
        theta = inputs['theta'] #parameters for theta  sampling

        if ( theta_sampling == 'random_morphing_thetas' ): 
            tuple_0 = theta_sampling['prior']['parameter_0'] #tuple for parameter 0
            tuple_1 = theta_sampling['prior']['parameter_1'] #tuple for parameter 1
            prior = [ (str(tuple_0['prior_shape']), float(tuple_0['prior_param_0']), float(tuple_0['prior_param_1'])), \
                      (str(tuple_1['prior_shape']), float(tuple_1['prior_param_0']), float(tuple_1['prior_param_1']))  ] 
            
            x, theta0, theta1, y, r_xz, t_xz = extract_samples_train_local(
                theta=eval(theta_sampling)(theta['n_thetas'], prior),
                n_samples=int(inputs['n_samples']['train']),
                test_split=test_split,
                folder='/home/data/Samples_'+str(i),
                filename='train'      
            )

        else:
            x, theta0, theta1, y, r_xz, t_xz = sa.extract_samples_train_local(
                theta=eval(theta_sampling)(theta['argument']),
                n_samples=int(inputs['n_samples']['train']),
                test_split=test_split,
                folder='/home/data/Samples_'+str(i),
                filename='train'
            )

    #TRAIN GLOBAL
    if method in ['scandal']:
        theta_sampling = inputs['theta']['sampling_method']
        theta = inputs['theta'] #parameters for theta sampling

        if ( theta_sampling == 'random_morphing_thetas' ): 
            tuple_0 = theta_sampling['prior']['parameter_0'] #tuple for parameter 0
            tuple_1 = theta_sampling['prior']['parameter_1'] #tuple for parameter 1
            prior = [ (str(tuple_0['prior_shape']), float(tuple_0['prior_param_0']), float(tuple_0['prior_param_1'])), \
                      (str(tuple_1['prior_shape']), float(tuple_1['prior_param_0']), float(tuple_1['prior_param_1']))  ] 
            
            x, theta0, theta1, y, r_xz, t_xz = extract_samples_train_global(
                theta=eval(theta_sampling)(theta['n_thetas'], prior),
                n_samples=int(inputs['n_samples']['train']),
                test_split=test_split,
                folder='/home/data/Samples_'+str(i),
                filename='train'      
            )

        else:
            x, theta0, theta1, y, r_xz, t_xz = sa.extract_samples_train_global(
                theta=eval(theta_sampling)(theta['argument']),
                n_samples=int(inputs['n_samples']['train']),
                test_split=test_split,
                folder='/home/data/Samples_'+str(i),
                filename='train'
            )

    