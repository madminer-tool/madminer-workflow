#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import h5py
import logging
import numpy as np
import os
import sys
import yaml
from pathlib import Path

from madminer.fisherinformation import FisherInformation
from madminer.limits import AsymptoticLimits
from madminer.ml import LikelihoodEstimator
from madminer.ml import ParameterizedRatioEstimator
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

model_dir = str(project_dir.joinpath('models'))
rates_dir = str(project_dir.joinpath('rates'))
results_dir = str(project_dir.joinpath('results'))
tests_dir = str(project_dir.joinpath('test'))

sampling_methods = {
    'benchmark': benchmark,
    'benchmarks': benchmarks,
    'morphing_points': morphing_point,
    'random_morphing_points': random_morphing_points,
}


############################
##### Argument parsing #####
############################

inputs_file = sys.argv[1]
eval_folder = sys.argv[2]
config_file = sys.argv[3]

with open(inputs_file) as f:
    inputs = yaml.safe_load(f)


#############################
### Configuration parsing ###
#############################

asymptotic = dict(inputs['asymptotic_limits'])
fisher_info = dict(inputs['fisher_information'])
gen_method = str(os.path.split(os.path.abspath(eval_folder))[1])
luminosity = float(inputs['luminosity'])
test_split = float(inputs['test_split'])
num_samples = int(inputs['n_samples']['train'])

with h5py.File(config_file, 'r') as f:
    parameters = f['parameters']['names']


###############################
## Define calc. events func. ##
###############################

def calc_num_events(config_file, lum):
    """
    Calculate the number of events
    :param config_file: path to the configuration file
    :param lum: luminosity value
    :return: int
    """

    # First: print limits debug information
    limits = AsymptoticLimits(config_file)

    xs_limits = limits._calculate_xsecs(thetas=[theta_true], test_split=test_split)
    logging.info(f"Asymptotic limits: {xs_limits[0]}")

    # Second: print samples debug information
    sample_augmenter = SampleAugmenter(config_file, include_nuisance_parameters=False)

    _, xs_sa, _ = sample_augmenter.cross_sections(theta=morphing_point(theta_true))
    logging.info(f"SampleAugmenter cross sections: {xs_sa[0]}")

    _, weights = sample_augmenter.weighted_events(theta='sm')
    logging.info(f"SampleAugmenter weighted events: {sum(weights)}")

    xs_xsecs, _ = sample_augmenter.xsecs(
        thetas=[theta_true],
        partition='train',
        test_split=test_split
    )
    logging.info(f"SampleAugmenter xsecs: ", xs_xsecs[0])

    # Second: calculate expected number of events
    _, xs, _ = sample_augmenter.cross_sections(theta=morphing_point(theta_true))
    return lum * xs[0]


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


##############################
# Define data gen. functions #
##############################

def generate_test_data_ratio(method, params):
    """
    Generates test data files given a particular method (ratio)
    :param method: name of the MadMiner method to generate theta
    :param params: list of parameter names the analysis is taking
    """

    sampler = SampleAugmenter(config_file, include_nuisance_parameters=False)
    thetas = inputs['evaluation'][method]

    if len(thetas) == 1:
        theta = thetas['theta']
        theta_sampling = theta['sampling_method']

        # Default arguments
        theta_args = theta['argument']

        # Overriding default 'theta' arguments
        if theta_sampling == 'random_morphing_points':
            theta_args = generate_theta_args(theta, params)

        # Getting the specified sampling method
        theta_method = sampling_methods.get(theta_sampling)

        sampler.sample_test(
            theta=theta_method(*theta_args),
            n_samples=num_samples,
            folder=tests_dir + f'/{method}/',
            filename='test',
        )

    elif len(thetas) == 2:
        theta_0 = thetas['theta_0']
        theta_1 = thetas['theta_1']
        theta_0_sampling = theta_0['sampling_method']
        theta_1_sampling = theta_1['sampling_method']

        # Default arguments
        theta_0_args = [theta_0['argument']]
        theta_1_args = [theta_1['argument']]

        # Overriding default 'theta' arguments
        if theta_0_sampling == 'random_morphing_points':
            theta_0_args = generate_theta_args(theta_0, parameters)

        # Overriding default 'theta' arguments
        if theta_1_sampling == 'random_morphing_points':
            theta_1_args = generate_theta_args(theta_1, parameters)

        # Getting the specified sampling method
        theta_0_method = sampling_methods.get(theta_0_sampling)
        theta_1_method = sampling_methods.get(theta_1_sampling)

        sampler.sample_train_ratio(
            theta0=theta_0_method(*theta_0_args),
            theta1=theta_1_method(*theta_1_args),
            n_samples=num_samples,
            folder=tests_dir + f'/{method}/',
            filename='test'
        )


def generate_test_data_score(method, params):
    """
    Generates test data files given a particular method (score)
    :param method: name of the MadMiner method to generate theta
    :param params: list of parameter names the analysis is taking
    """

    sampler = SampleAugmenter(config_file, include_nuisance_parameters=False)
    thetas = inputs['evaluation'][method]

    theta = thetas['theta']
    theta_sampling = theta['sampling_method']

    # Default arguments
    theta_args = theta['argument']

    # Overriding default 'theta' arguments
    if theta_sampling == 'random_morphing_points':
        theta_args = generate_theta_args(theta, params)

    # Getting the specified sampling method (defaults to 'benchmark')
    theta_method = sampling_methods.get(theta_sampling, benchmark)

    sampler.sample_train_local(
        theta=theta_method(*theta_args),
        n_samples=num_samples,
        folder=tests_dir + f'/{method}/',
        filename='test',
    )


##############################
# Define args override func. #
##############################

def save_limits(mode, method, models_path, include_xs):
    """
    Generates and save the expected limits
    :param mode: {'ml', 'histo'}
    :param method: method used to generate theta values
    :param models_path: path to where the models are
    :param include_xs: flag indicating whether or not include the cross section
    """

    _, p_values, best_fit_index = limits.expected_limits(
        mode=mode,
        theta_true=theta_true,
        grid_ranges=theta_ranges,
        grid_resolutions=resolutions,
        model_file=models_path + '/' + method,
        include_xsec=include_xs,
        luminosity=luminosity,
    )

    results_path = results_dir + f'/{method}/{mode}'
    os.makedirs(results_path, exist_ok=True)

    if include_xs:
        np.save(file=results_path + f'/{method}.npy', arr=[p_values, best_fit_index])
    else:
        np.save(file=results_path + f'/{method}_kin.npy', arr=[p_values, best_fit_index])


###############################
## Evaluate asymptotic limit ##
###############################

score_estimator_methods = {'sally', 'sallino'}
ratio_estimator_methods = {'alice', 'alices', 'cascal', 'carl', 'rolr', 'rascal'}

if asymptotic['bool']:
    theta_ranges = []

    for asymp_theta in asymptotic['region'].keys():
        theta_min, theta_max = asymptotic['region'][asymp_theta]
        theta_ranges.append((theta_min, theta_max))

    print(f'Theta range: {theta_ranges}')

    include_xsec = asymptotic['include_xsec']
    resolutions = asymptotic['resolutions']
    theta_true = asymptotic['theta_true']
    hist_vars = asymptotic['hist_vars']


    # Computes rates and grid
    limits = AsymptoticLimits(config_file)
    theta_grid, p_values, best_fit_index = limits.expected_limits(
        mode="rate",
        theta_true=theta_true,
        grid_ranges=theta_ranges,
        grid_resolutions=resolutions,
        include_xsec=True,
        luminosity=luminosity,
    )

    np.save(file=rates_dir + '/grid.npy', arr=theta_grid)
    np.save(file=rates_dir + '/rate.npy', arr=[p_values, best_fit_index])


    # Compute cross sections and effective samples
    augmenter = SampleAugmenter(config_file, include_nuisance_parameters=False)
    cross_sec_grid = []
    num_effect_grid = []
    num_sample_test = 10000

    for theta_elem in theta_grid:
        _, cross_secs, _ = augmenter.cross_sections(theta=morphing_point(theta_elem))
        _, _, num_effect = augmenter.sample_train_plain(
            theta=morphing_point(theta_elem),
            n_samples=num_sample_test,
        )
        cross_sec_grid.append(cross_secs)
        num_effect_grid.append(num_effect / float(num_sample_test))

    np.save(file=rates_dir + '/xs_grid.npy', arr=np.array(cross_sec_grid))
    np.save(file=rates_dir + '/neff_grid.npy', arr=np.array(num_effect_grid))


    # Compute and save histogram
    for flag in include_xsec:
        _ , p_values, best_fit_index = limits.expected_limits(
            mode="histo",
            theta_true=theta_true,
            grid_ranges=theta_ranges,
            grid_resolutions=resolutions,
            hist_vars=[hist_vars],
            include_xsec=flag,
            luminosity=luminosity
        )

        if flag:
            np.save(file=rates_dir + '/histo.npy', arr=[p_values, best_fit_index])
        else:
            np.save(file=rates_dir + '/histo_kin.npy', arr=[p_values, best_fit_index])

    # Compute and save ML or Histogram limits
    for flag in include_xsec:
        if gen_method in ratio_estimator_methods:
            save_limits(mode='ml', method=gen_method, models_path=eval_folder, include_xs=flag)
        elif gen_method in score_estimator_methods:
            save_limits(mode='histo', method=gen_method, models_path=eval_folder, include_xs=flag)
        else:
            raise ValueError('Invalid generation method')


###############################
#### Evaluate fisher info. ####
###############################

if fisher_info['bool'] and gen_method == 'sally':
    fisher = FisherInformation(config_file, include_nuisance_parameters=False)
    theta_true = [
        float(fisher_info['theta_true'][0]),
        float(fisher_info['theta_true'][1]),
        float(fisher_info['theta_true'][2]),
    ]

    fisher.calculate_fisher_information_full_detector(
        theta=[0., 0., 0.],
        model_file=model_dir + '/sally/sally',
        luminosity=30000.
    )


###############################
### Evaluate: train VS test ###
###############################

# Generate test data and num_events
generate_test_data_ratio(gen_method, parameters)
num_events = calc_num_events(config_file, luminosity)

# Get Log Likelihood Ratio metrics
llr_raw = []
llr_rescaled = []
thetas = []
scores = []

theta_grid = np.load(rates_dir + '/grid.npy')


if gen_method in ratio_estimator_methods:
    forge = ParameterizedRatioEstimator()
    eval_func = forge.evaluate_log_likelihood_ratio
    eval_flag = True

elif gen_method in score_estimator_methods:
    forge = LikelihoodEstimator()
    eval_func = forge.evaluate_log_likelihood
    eval_flag = False

else:
    raise ValueError('Invalid generation method')


forge.load(eval_folder + '/' + gen_method)

for theta_elem in theta_grid:
    llr, score = eval_func(
        x=tests_dir + f'/{gen_method}/x_test.npy',
        theta=np.array([theta_elem]),
        test_all_combinations=True,
        evaluate_score=eval_flag,
    )

    llr_raw = sum(llr[0]) / num_sample_test
    llr_rescaled = num_events * llr_raw

    llr_raw.append(llr_raw)
    llr_rescaled.append(llr_rescaled)
    thetas.append(theta_elem)
    scores.append(score)


################################
## Save Log Like. Ratio files ##
################################

limits = AsymptoticLimits(config_file)
llr_sub, _ = limits._subtract_mle(llr_rescaled)
p_values = limits.asymptotic_p_value(llr_sub)

llr_raw_file = results_dir + f'/{gen_method}/llr_raw.npy'
np.save(file=llr_raw_file, arr=llr_raw)
print(f'Saved Raw mean -2 log R to file: {llr_raw_file}')

llr_rescaled_file = results_dir + f'/{gen_method}/llr_rescaled.npy'
np.save(file=llr_rescaled_file, arr=llr_rescaled)
print(f'Saved rescaled -2 log R to file: {llr_rescaled_file}')

llr_subtracted_file = results_dir + f'/{gen_method}/llr_subtracted.npy'
np.save(file=llr_subtracted_file, arr=llr_sub)
print(f'Saved subtracted -2 log R to file: {llr_subtracted_file}')

pvals_file = results_dir + f'/{gen_method}/p_values.npy'
np.save(file=pvals_file, arr=p_values)
print(f'Saved p-values to file: {pvals_file}')
