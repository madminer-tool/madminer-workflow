from __future__ import absolute_import, division, print_function, unicode_literals

#import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
#%matplotlib inline

import yaml
import sys

from madminer.core import MadMiner
from madminer.delphes import DelphesProcessor
from madminer.sampling import combine_and_shuffle
from madminer.sampling import SampleAugmenter
from madminer.sampling import constant_benchmark_theta, multiple_benchmark_thetas, random_morphing_thetas,constant_morphing_theta
from madminer.ml import MLForge
from madminer.plotting import plot_2d_morphing_basis, plot_distributions

#tunning for evaluation
inputs_file = sys.argv[1] 
with open(inputs_file) as f:
    inputs = yaml.safe_load(f)

method = str(inputs['method'])

#folder with trained files
eval_folder_path = str(sys.argv[2]) 

#configurate file 
h5_file = sys.argv[3]


#create test sample
sa = SampleAugmenter(h5_file)   #'data/madminer_example_shuffled.h5'

test=inputs['test_samples']
_ = sa.extract_samples_test(
    theta=eval(test['sampling_method'])(test['argument']),
    n_samples=int(test['nsamples']), #change too
    folder='/home/data/test',
    filename='test'
)

#evaluate
forge = MLForge() #Estimator() v.0.3
forge.load(eval_folder_path+'/'+method)  #'methods/alices'


#?
theta_denom = np.array([[0.,0.]])
np.save('/home/data/test/theta_ref.npy', theta_denom)


#perform the test + evaluation score acconding to method
if( method  in ['alice',  'alices',  'carl',  'nde', 'rascal',  'rolr',  'scandal'] ):
	
	#make and save grid
	evaluation = inputs['evaluation']['theta_each']
	theta_each = np.linspace( float(evaluation['start']), float(evaluation['stop']), int(evaluation['num']) ) 
	theta0, theta1 = np.meshgrid(theta_each, theta_each)
	theta_grid = np.vstack((theta0.flatten(), theta1.flatten())).T #numtidim
	np.save('/home/data/test/theta_grid.npy', theta_grid)

	log_r_hat, score_theta0, _ = forge.evaluate(
	    theta0_filename='/home/data/test/theta_grid.npy',
	    x='/home/data/test/x_test.npy',
	    evaluate_score=inputs['evaluation']['evaluate_score']
	)
	with open('/home/data/test/log_r_hat_'+method+'.npy', "w+") as f: #create file
		np.save(file='/home/data/test/log_r_hat_'+method, arr=log_r_hat)
	
	with open('/home/data/test/score_theta0_'+method+'.npy', "w+") as g: #create file
		np.save(file='/home/data/test/score_theta0_'+method, arr=score_theta0)
	

	#plots
	if( bool(inputs['plots']['activate']) ):
		
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
		plt.savefig('/home/plots/expected_llr_'+method+'.png')	


if( method  in ['alice2', 'alices2', 'carl2', 'rascal2', 'rolr2' ] ):
	print('evaluation for this method is not yet implemented')
	pass 

	#make and save grid
	evaluation = inputs['evaluation']['theta_each']
	theta_each = np.linspace( float(evaluation['start']), float(evaluation['stop']), int(evaluation['num']) ) 
	theta0, theta1 = np.meshgrid(theta_each, theta_each)
	theta_grid = np.vstack((theta0.flatten(), theta1.flatten())).T #numtidim
	np.save('/home/data/test/theta_grid.npy', theta_grid)

	log_r_hat, score_theta0, score_theta1 = forge.evaluate(
	    theta0_filename='/home/data/test/theta_grid.npy',
	    theta1_filename='/home/data/test/theta_grid.npy', #TO DO !
	    x='/home/data/test/x_test.npy',
	    evaluate_score=False
	)
	with open('/home/data/test/log_r_hat_'+method+'.npy', "w+") as f: #create file
		np.save(file='/home/data/test/log_r_hat_'+method, arr=log_r_hat)		


if( method  in ['sally', 'sallino'] ):
	t_hat = forge.evaluate(
    	x='/home/data/samples/x_test.npy'
	)
	with open('/home/data/test/t_hat_'+method+'.npy', "w+") as f: #create file
		np.save(file='/home/data/test/t_hat_'+method, arr=t_hat)

	#plots
	if( bool(inputs['plots']['activate']) ):
		x = np.load('data/samples/x_test.npy')
		fig = plt.figure(figsize=(10,4))

		for i in range(2):
			ax = plt.subplot(1,2,i+1)
			sc = plt.scatter(x[::10,0], x[::10,1], c=t_hat[::10,i], s=10., cmap='viridis', vmin=-0.8, vmax=0.4)
			cbar = plt.colorbar(sc)
			cbar.set_label(r'$\hat{t}_' + str(i) + r'(x | \theta_{ref})$')
			plt.xlabel(r'$p_{T,j1}$ [GeV]')
			plt.ylabel(r'$\Delta \phi_{jj}$')
			plt.xlim(10.,400.)
			plt.ylim(-3.15,3.15)
			plt.tight_layout()
			plt.savefig('/home/plots/t_hat_'+method+'.png')	


#Fisher ifnormation
if( bool(inputs['fisher_information']['activate']) and method in ['sally'] ):
	fisher = inputs['fisher_information']
	fisher = FisherInformation(h5_file)

	fisher_information, _ = fisher.calculate_fisher_information_full_detector(
	    theta=fisher['theta'],
	    model_file=eval_folder_path+'/sally',
	    unweighted_x_sample_file='/home/data/samples/x_test.npy',
	    luminosity=float(fisher['luminosity'])
	)
	print('Kinematic Fisher information after {} ifb:\n{}'.format(float(fisher['luminosity']), fisher_information))

	#plots
	if( bool(inputs['plots']['activate']) ):
		#1
		plot_fisher_contour = plot_fisher_information_contours_2d(
		    [fisher_information],
		    xrange=(-1,1),
		    yrange=(-1,1)
		)
		plot_fisher_contour.savefig('/home/plots/plot_fisher_contour.png')
		
		#2
		print('plot_distribution_of_information yer to be implemented')
		pass
		plot_fisher_distr = plot_distribution_of_information(
			[fisher_information_matrices],
			xbins=None, 
			xsecs=None,
		)
		
		plot_fisher_distr.savefig('/home/plots/plot_fisher_distr.png')