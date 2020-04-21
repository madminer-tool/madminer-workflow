#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import h5py
import logging
import sys
import yaml
from pathlib import Path

from madminer.sampling import combine_and_shuffle
from madminer.sampling import SampleAugmenter

# These methods are applied if specified in the input files
from madminer.sampling import benchmark
from madminer.sampling import benchmarks
from madminer.sampling import morphing_point
from madminer.sampling import random_morphing_points


##########################
##### Set up logging #####
##########################

logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.INFO
)

for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)


############################
##### Global variables #####
############################

project_dir = Path(__file__).parent.parent
data_dir = str(project_dir.joinpath('data'))

sampling_methods = {
    'benchmark': benchmark,
    'benchmarks': benchmarks,
    'morphing_points': morphing_point,
    'random_morphing_points': random_morphing_points,
}


############################
##### Argument parsing #####
############################

num_train_samples = int(sys.argv[1])
data_file = str(sys.argv[2])
inputs_file = str(sys.argv[3])

with open(inputs_file) as f:
    inputs = yaml.safe_load(f)


#############################
### Configuration parsing ###
#############################

nuisance = inputs['include_nuisance_parameters']
methods = inputs['methods']
shuffle = inputs['shuffle']
samples = int(inputs['n_samples']['train'])
split = float(inputs['test_split'])

with h5py.File(data_file, 'r') as f:
    parameters = f['parameters']['names']


#############################
#### Instantiate Sampler ####
#############################

if shuffle:
    data_file_shuffled = data_dir + '/' + 'combined_delphes_shuffled.h5'
    combine_and_shuffle(input_filenames=[data_file], output_filename=data_file_shuffled)
    sampler = SampleAugmenter(data_file_shuffled, include_nuisance_parameters=nuisance)
else:
    sampler = SampleAugmenter(data_file, include_nuisance_parameters=nuisance)


##############################
# Define args override func. #
##############################

def generate_theta_args(theta_spec, params):
    """
    Generates the theta arguments that the method will take later on
    :param theta_spec: theta specification on the inputs file
    :param params: list of parameter names the analysis is taking
    :return: list
    """

    prior = []

    for p, _ in enumerate(params):
        param_prior = theta_spec['prior'][f'parameter_{p}']
        prior.append(
            (
                param_prior['prior_shape'],
                float(param_prior['prior_param_0']),
                float(param_prior['prior_param_1']),
            )
        )

    return [theta_spec['n_thetas'], prior]


#############################
## Create training samples ##
#############################

# Different methods have different arguments
train_ratio_methods = {'alice', 'alices', 'cascal', 'carl', 'rolr', 'rascal'}
train_local_methods = {'sally', 'sallino'}
train_global_methods = {'scandal'}

# Iterate through the methods
for method in methods:

    training_params = inputs[method]
    print(f'Sampling from method: {method}')


    for i in range(num_train_samples):

        if method in train_ratio_methods:
            theta_0 = training_params['theta_0']
            theta_1 = training_params['theta_1']
            theta_0_sampling = theta_0['sampling_method']
            theta_1_sampling = theta_1['sampling_method']

            # Default arguments in case no theta is 'random_morphing_points'
            theta_0_args = ['w']
            theta_1_args = ['sm']

            # Overriding default 'theta' arguments
            if theta_0_sampling == 'random_morphing_points':
                theta_0_args = generate_theta_args(theta_0, parameters)
                theta_1_args = [theta_1['argument']]

            # Overriding default 'theta' arguments
            if theta_1_sampling == 'random_morphing_points':
                theta_0_args = [theta_0['argument']]
                theta_1_args = generate_theta_args(theta_1, parameters)

            # Getting the specified sampling method (defaults to 'benchmark')
            theta_0_method = sampling_methods.get(theta_0_sampling, benchmark)
            theta_1_method = sampling_methods.get(theta_1_sampling, benchmark)

            sampler.sample_train_ratio(
                theta0=theta_0_method(*theta_0_args),
                theta1=theta_1_method(*theta_1_args),
                n_samples=samples,
                folder=data_dir + f'/Samples_{method}_{i}',
                filename=method + '_train',
                test_split=split,
            )


        elif method in train_local_methods:
            theta = training_params['theta']
            theta_sampling = theta['sampling_method']

            # Default arguments in case theta is 'random_morphing_points'
            theta_args = [theta['argument']]

            # Overriding default 'theta' arguments
            if theta_sampling == 'random_morphing_points':
                theta_args = generate_theta_args(theta, parameters)

            # Getting the specified sampling method (defaults to 'benchmark')
            theta_method = sampling_methods.get(theta_sampling, benchmark)

            sampler.sample_train_local(
                theta=theta_method(*theta_args),
                n_samples=samples,
                folder=data_dir + f'/Samples_{method}_{i}',
                filename=method + '_train',
                test_split=split,
            )


        elif method in train_global_methods:
            theta = training_params['theta']
            theta_sampling = theta['sampling_method']

            # Default arguments in case theta is 'random_morphing_points'
            theta_args = [theta['argument']]

            # Overriding default 'theta' arguments
            if theta_sampling == 'random_morphing_points':
                theta_args = generate_theta_args(theta, parameters)

            # Getting the specified sampling method (defaults to 'benchmark')
            theta_method = sampling_methods.get(theta_sampling, benchmark)

            sampler.sample_train_density(
                theta=theta_method(*theta_args),
                n_samples=samples,
                folder=data_dir + f'/Samples_{method}_{i}',
                filename=method + '_train',
                test_split=split,
            )


        else:
            raise ValueError('Invalid sampling method')
