#!/usr/bin/python

import logging
import matplotlib
import math
import numpy as np
import sys
import yaml
from matplotlib import pyplot as plt
from pathlib import Path


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

method_colors = {
    'alice': 'C1',
    'alices': 'red',
    'rascal': 'magenta',
    'sally': 'C0',
}


############################
##### Argument parsing #####
############################

inputs_file = sys.argv[1]
output_dir = Path(sys.argv[2])

model_dir = str(output_dir.joinpath('models'))
plots_dir = str(output_dir.joinpath('plots'))
rates_dir = str(output_dir.joinpath('rates'))
results_dir = str(output_dir.joinpath('results'))
tests_dir = str(output_dir.joinpath('test'))

with open(inputs_file) as f:
    inputs = yaml.safe_load(f)


#############################
### Configuration parsing ###
#############################

methods = list(inputs['methods'])
plotting = dict(inputs['plotting'])
plotting_method = str(plotting['all_methods_pvalue'])

asymptotic = dict(inputs['asymptotic_limits'])
resolutions = list(asymptotic['resolutions'])
asymp_region = dict(asymptotic['region'])
theta_0_min, theta_0_max = asymp_region['theta0_min_max']
theta_1_min, theta_1_max = asymp_region['theta1_min_max']

matplotlib.use('Agg')


##############################
## Define theta props func. ##
##############################

def build_theta_props(theta_min, theta_max, resolution):
    """
    Computes theta size, edges and centers
    :param theta_min:
    :param theta_max:
    :param resolution:
    :return: int, list, list
    """

    theta_size = (theta_max - theta_min) / (resolution - 1)

    theta_edges = np.linspace(
        start=theta_min - (theta_size / 2),
        stop=theta_max + (theta_size / 2),
        num=resolution + 1,
    )

    theta_centers = np.linspace(
        start=theta_min,
        stop=theta_max,
        num=resolution,
    )

    return theta_size, theta_edges, theta_centers


#############################
### Define plotting func. ###
#############################

def do_plot(
    p_values_expected,
    best_fit_expected,
    colors,
    styles,
    titles,
    n_cols=3,
    theta0range=(-1, 1),
    theta1range=(-1, 1),
    resolution=10
):

    # Get bin sizes and positions
    theta_0_size, theta_0_edges, theta_0_centers = build_theta_props(
        theta_min=theta0range[0],
        theta_max=theta0range[1],
        resolution=resolution,
    )

    theta_1_size, theta_1_edges, theta_1_centers = build_theta_props(
        theta_min=theta1range[0],
        theta_max=theta1range[1],
        resolution=resolution,
    )

    c_min, c_max = 1.e-3, 1.

    # Preparing plot
    n_methods = len(p_values_expected) + 1
    n_rows = (n_methods + n_cols - 1) // n_cols
    figure = plt.figure(figsize=(6.0 * n_cols, 5.0 * n_rows))

    for index, _ in enumerate(p_values_expected):

        # Panel
        ax = plt.subplot(n_rows, n_cols, index + 1)

        # p-value
        pcm = ax.pcolormesh(
            theta_0_edges,
            theta_1_edges,
            p_values_expected[index].reshape((resolution, resolution)),
            norm=matplotlib.colors.LogNorm(vmin=c_min, vmax=c_max),
            cmap='Greys_r',
        )

        cbar = figure.colorbar(pcm, ax=ax, extend='both')
        cbar.set_label('Expected p-value')

        # Contours
        plt.contour(
            theta_0_centers,
            theta_0_centers,
            p_values_expected[index].reshape((resolution, resolution)),
            levels=[0.61],
            linestyles=styles[index],
            colors=colors[index],
            label=r"$1\sigma$ contour"
        )

        # Best fit
        plt.scatter(
            variables_to_plot['theta_grid'][best_fit_expected[index]][0],
            variables_to_plot['theta_grid'][best_fit_expected[index]][1],
            s=80.,
            color=colors[index],
            marker='+',
            label="Best Fit"
        )    

        # Title
        plt.title(titles[index])
        plt.xlabel(r'$\theta_0$')
        plt.ylabel(r'$\theta_1$')
        plt.legend()

    plt.tight_layout()
    plt.savefig(plots_dir + '/all_methods_separate.png')


#############################
## Load previous rate data ##
#############################

loaded_grid_data = np.load(rates_dir + '/grid.npy', allow_pickle=True)
loaded_rate_data = np.load(rates_dir + '/rate.npy', allow_pickle=True)
loaded_histo_data = np.load(rates_dir + '/histo.npy', allow_pickle=True)
loaded_histo_kin_data = np.load(rates_dir + '/histo_kin.npy', allow_pickle=True)

variables_to_plot = {
    'theta_grid': loaded_grid_data,
    'p_values_expected_rate': loaded_rate_data[0],
    'best_fit_expected_rate': loaded_rate_data[1],
    'p_values_expected_histo': loaded_histo_data[0],
    'best_fit_expected_histo': loaded_histo_data[1],
    'p_values_expected_histo_kin': loaded_histo_kin_data[0],
    'best_fit_expected_histo_kin': loaded_histo_kin_data[1],
}


#############################
### Load previous results ###
#############################

ml_folder_methods = {'alice', 'alices', 'cascal', 'carl', 'rolr', 'rascal'}
hg_folder_methods = {'sally', 'sallino'}

for method in methods:

    if method in ml_folder_methods:
        a = np.load(results_dir + f'/{method}/ml/{method}.npy', allow_pickle=True)
        b = np.load(results_dir + f'/{method}/ml/{method}_kin.npy', allow_pickle=True)
    elif method in hg_folder_methods:
        a = np.load(results_dir + f'/{method}/histo/{method}.npy', allow_pickle=True)
        b = np.load(results_dir + f'/{method}/histo/{method}_kin.npy', allow_pickle=True)
    else:
        raise ValueError('Invalid method')

    variables_to_plot['p_values_expected_' + method] = a[0]
    variables_to_plot['best_fit_expected_' + method] = a[1]
    variables_to_plot['p_values_expected_' + method + '_kin'] = b[0]
    variables_to_plot['best_fit_expected_' + method + '_kin'] = b[1]


#############################
##### Plot all together #####
#############################

if plotting['all_methods']:

    # Get bin sizes and positions
    theta_0_size, theta_0_edges, theta_0_centers = build_theta_props(
        theta_min=theta_0_min,
        theta_max=theta_0_max,
        resolution=resolutions[0],
    )

    theta_1_size, theta_1_edges, theta_1_centers = build_theta_props(
        theta_min=theta_1_min,
        theta_max=theta_1_max,
        resolution=resolutions[1],
    )

    # Define plot
    fig = plt.figure(figsize=(6, 5))
    ax = plt.gca()

    # Alices dependent
    c_min, c_max = 1.e-3, 1.

    # P-values keys
    p_values_key = f'p_values_expected_{plotting_method}_kin'

    print(f"Theta 0 edges shapes: {theta_0_edges.shape}")
    print(f"Theta 1 edges shapes: {theta_1_edges.shape}")
    print(f"P-values shapes: {variables_to_plot[p_values_key].shape}")

    pcm = ax.pcolormesh(
        theta_0_edges,
        theta_1_edges,
        variables_to_plot[p_values_key].reshape([resolutions[0], resolutions[1]]),
        norm=matplotlib.colors.LogNorm(vmin=c_min, vmax=c_max),
        cmap='Greys_r'
    )

    c_bar = fig.colorbar(pcm, ax=ax, extend='both')

    for method in methods:
        print(f'Plotting: {method}')
        color = method_colors.get(method)

        plt.contour(
            theta_0_centers,
            theta_0_centers,
            variables_to_plot['p_values_expected_' + method].reshape([resolutions[0], resolutions[1]]),
            levels=[0.61],
            linestyles='-',
            colors=color,
        )
        plt.contour(
            theta_0_centers,
            theta_0_centers,
            variables_to_plot['p_values_expected_' + method + '_kin'].reshape([resolutions[0], resolutions[1]]),
            levels=[0.61],
            linestyles='--',
            colors=color,
        )

        plt.scatter(
            variables_to_plot['theta_grid'][variables_to_plot['best_fit_expected_' + method]][0],
            variables_to_plot['theta_grid'][variables_to_plot['best_fit_expected_' + method]][1],
            s=80.,
            color=color,
            marker='*',
            label=method.upper(),
        )
        plt.scatter(
            variables_to_plot['theta_grid'][variables_to_plot['best_fit_expected_' + method + '_kin']][0],
            variables_to_plot['theta_grid'][variables_to_plot['best_fit_expected_' + method + '_kin']][1],
            s=80.,
            color=color,
            marker='+',
            label=(method+'-kin').upper(),
        )

    # Plot rates
    plt.contour(
        theta_0_centers,
        theta_0_centers,
        variables_to_plot['p_values_expected_rate'].reshape([resolutions[0], resolutions[1]]),
        levels=[0.61],
        linestyles='-',
        colors='black',
    )
    plt.scatter(
        variables_to_plot['theta_grid'][variables_to_plot['best_fit_expected_rate']][0],
        variables_to_plot['theta_grid'][variables_to_plot['best_fit_expected_rate']][1],
        s=80.,
        color='black',
        marker='*',
        label="xsec",
    )

    # Plot histogram
    plt.contour(
        theta_0_centers,
        theta_0_centers,
        variables_to_plot['p_values_expected_histo'].reshape([resolutions[0], resolutions[1]]),
        levels=[0.61],
        linestyles='-',
        colors='limegreen',
        label="Histo",
    )
    plt.scatter(
        variables_to_plot['theta_grid'][variables_to_plot['best_fit_expected_histo']][0],
        variables_to_plot['theta_grid'][variables_to_plot['best_fit_expected_histo']][1],
        s=80.,
        color='limegreen',
        marker='*',
        label="Histo",
    )

    # Plot kin
    plt.contour(
        theta_0_centers,
        theta_0_centers,
        variables_to_plot['p_values_expected_histo_kin'].reshape([resolutions[0], resolutions[1]]),
        levels=[0.61],
        linestyles='--',
        colors='limegreen',
    )
    plt.scatter(
        variables_to_plot['theta_grid'][variables_to_plot['best_fit_expected_histo_kin']][0],
        variables_to_plot['theta_grid'][variables_to_plot['best_fit_expected_histo_kin']][1],
        s=80.,
        color='limegreen',
        marker='+',
        label="Histo-Kin",
    )

    # Finish plot
    plt.legend()
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    c_bar.set_label(f'Expected p-value ({plotting_method.upper()}-Kinematics)')

    plt.tight_layout()
    plt.savefig(plots_dir + '/all_methods.png')


#############################
###### Plot separately ######
#############################

if plotting['all_methods_separate']:

    plot_input = np.array([
        [variables_to_plot['p_values_expected_rate'], variables_to_plot['best_fit_expected_rate'], "Rate", "black", "solid"],
        [variables_to_plot['p_values_expected_histo'], variables_to_plot['best_fit_expected_histo'], "Histo", "limegreen", "solid"],
        [variables_to_plot['p_values_expected_histo_kin'], variables_to_plot['best_fit_expected_histo_kin'], "Histo-Kin", "limegreen", "dashed"],
        [variables_to_plot['p_values_expected_sally'], variables_to_plot['best_fit_expected_sally'], "SALLY", "C0", "solid"],
        [variables_to_plot['p_values_expected_sally_kin'], variables_to_plot['best_fit_expected_sally_kin'], "SALLY-Kin", "C0", "dashed"],
        [variables_to_plot['p_values_expected_alices'], variables_to_plot['best_fit_expected_alices'], "ALICES", "red", "solid"],
        [variables_to_plot['p_values_expected_alices_kin'], variables_to_plot['best_fit_expected_alices_kin'], "ALICES-Kin", "red","dashed"],
    ])

    do_plot(
        p_values_expected=plot_input[:, 0],
        best_fit_expected=plot_input[:, 1],
        colors=plot_input[:, 3],
        styles=plot_input[:, 4],
        titles=plot_input[:, 2],
        theta0range=[theta_0_min, theta_0_max],
        theta1range=[theta_1_min, theta_1_max],
        resolution=resolutions[0],
    )


#############################
##### Plot correlations #####
#############################

if plotting['correlations']:

    for method in inputs['plotting']['correlations_methods']:

        if len(inputs[method]) == 1:
            theta_file = 'theta_test.npy'
        elif len(inputs[method]) == 2:
            theta_file = 'theta0_test.npy'
        else:
            raise ValueError('Invalid number of evaluation methods')

        theta_test_path = tests_dir + f'/{method}/{theta_file}'
        r_truth_path = tests_dir + f'/{method}/r_xz_test.npy'
        x_test_path = tests_dir + f'/{method}/x_test.npy'

        # Joint LLR
        theta_test = np.load(theta_test_path)
        r_truth_test = np.load(r_truth_path)

        r_truth_test = r_truth_test.flatten()
        llr_truth_test = [math.log(r) for r in r_truth_test]
        llr_truth_test = np.array(llr_truth_test)

        # Estimated LLR
        from madminer.ml import ParameterizedRatioEstimator
        estimator_load = ParameterizedRatioEstimator()
        estimator_load.load(model_dir + f'/{method}/{method}')

        llr_ml_test, _ = estimator_load.evaluate_log_likelihood_ratio(
            x=x_test_path,
            theta=theta_test_path,
            test_all_combinations=False,
            evaluate_score=False,
        )

        # Define Colormap
        from matplotlib import cm
        from matplotlib.colors import ListedColormap
        old_colors = cm.get_cmap('Greens', 256)
        new_colors = old_colors(np.linspace(0, 1, 256))
        new_colors[:5, :] = np.array([1, 1, 1, 1])
        new_colormap = ListedColormap(new_colors)

        # Plot
        my_range = (-2, 2)
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(5, 5)

        ax.hist2d(
            llr_truth_test,
            llr_ml_test,
            bins=(40, 40),
            range=(my_range, my_range),
            cmap=new_colormap,
        )

        ax.set_xlabel('LLR: Truth')
        ax.set_ylabel('LLR: Estimated')

        plt.title(f'Correlation {method.upper()}')
        plt.tight_layout()
        plt.savefig(plots_dir + f'/correlation_{method.upper()}.png')
