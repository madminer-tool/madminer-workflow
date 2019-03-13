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

inputs_file = sys.argv[1]
with open(inputs_file) as f:
    inputs = yaml.safe_load(f)

model = str()

eval_folder_path = str(sys.argv[2])

#make and save grid
theta_each = np.linspace(-20.,20.,21) #input
theta0, theta1 = np.meshgrid(theta_each, theta_each)
theta_grid = np.vstack((theta0.flatten(), theta1.flatten())).T #numtidim
np.save('data/samples/theta_grid.npy', theta_grid)

theta_denom = np.array([[0.,0.]])
np.save('data/samples/theta_ref.npy', theta_denom)


#evaluate
forge = Estimator() #MLForge()
forge.load(eval_folder_path)  #'models/alices'

log_r_hat, _, _ = forge.evaluate(
    theta0_filename='/home/data/samples/theta_grid.npy',
    x='/home/data/samples/x_test.npy',
    evaluate_score=False
)

#log_r_hat save to file 


#plots
plots_bool = bool(inputs['plots'])

if(plots_bool):
	bin_size = theta_each[1] - theta_each[0]
	edges = np.linspace(theta_each[0] - bin_size/2, theta_each[-1] + bin_size/2, len(theta_each)+1)

	fig = plt.figure(figsize=(6,5))
	ax = plt.gca()

	expected_llr = np.mean(log_r_hat,axis=1)
	best_fit = theta_grid[np.argmin(-2.*expected_llr)]

	cmin, cmax = np.min(-2*expected_llr), np.max(-2*expected_llr)
	    
	pcm = ax.pcolormesh(edges, edges, -2. * expected_llr.reshape((21,21)),
	                    norm=matplotlib.colors.Normalize(vmin=cmin, vmax=cmax),
	                    cmap='viridis_r')
	cbar = fig.colorbar(pcm, ax=ax, extend='both')

	plt.scatter(best_fit[0], best_fit[1], s=80., color='black', marker='*')

	plt.xlabel(r'$\theta_0$')
	plt.ylabel(r'$\theta_1$')
	cbar.set_label(r'$\mathbb{E}_x [ -2\, \log \,\hat{r}(x | \theta, \theta_{SM}) ]$')

	plt.tight_layout()
	plt.savefig('/home/plots/llr.png')



#creates test samples JOHANN eval wf  or custom test
#MOVE TO evaluation.py
_ = sa.extract_samples_test(
    theta=constant_morphing_theta(str(inputs['benchmark'])),
    n_samples=int(inputs['n_samples']['test']), #change too
    folder='/home/data/Samples',
    filename='test'
)
