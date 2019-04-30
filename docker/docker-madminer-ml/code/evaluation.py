from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import os
import logging
import numpy as np
import math
import matplotlib
import glob
import yaml

from madminer.limits import AsymptoticLimits
from madminer.sampling import SampleAugmenter
from madminer.ml import ParameterizedRatioEstimator, ScoreEstimator, Ensemble
from madminer import sampling
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from madminer.fisherinformation import FisherInformation
from madminer.fisherinformation import project_information,profile_information

from madminer.plotting import plot_fisher_information_contours_2d
from madminer.plotting import plot_distributions

# logging
logging.basicConfig(
	format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
	datefmt='%H:%M',
	level=logging.INFO
)
for key in logging.Logger.manager.loggerDict:
	if "madminer" not in key:
		logging.getLogger(key).setLevel(logging.WARNING)


def func(x, sigmaSM, sigma0, sigma1, sigma00, sigma11, sigma01):
    return sigmaSM + x[0]*sigma0 + x[1]*sigma1 + x[0]**2*sigma00+ x[1]**2*sigma11 + x[0]*x[1]*sigma01

def funcrel(x, sigmaSM, sigma0, sigma1, sigma00, sigma11, sigma01):
    if x[0]==0 and x[1]==0:
        return 0
    else:
        return np.abs( (  x[0]**2*sigma00+ x[1]**2*sigma11 + x[0]*x[1]*sigma01)/(sigmaSM+x[0]*sigma0 + x[1]*sigma1 + x[0]**2*sigma00+ x[1]**2*sigma11 + x[0]*x[1]*sigma01 ) )
    
def funcrel2(x, sigmaSM, sigma0, sigma1, sigma00, sigma11, sigma01):
    if x[0]==0 and x[1]==0:
        return 0
    else:
        return ( abs(x[0]**2*sigma00)+ abs(x[1]**2*sigma11) + abs(x[0]*x[1]*sigma01)) / ( (sigmaSM+x[0]*sigma0) + abs(x[1]*sigma1 ) )




#tunning for evaluation
inputs_file = sys.argv[1] 
with open(inputs_file) as f:
	inputs = yaml.safe_load(f)

#folder with trained files
eval_folder_path = str(sys.argv[2]) 

#configurate file 
h5_file = sys.argv[3]

# get variables from inputs 
uselumi=float(inputs['uselumi'])

filename_path=glob.glob(eval_folder_path+'/'+'*.json')[0]

filename = filename_path.split("/", 3)[3]

method = str( filename.split("_", 3)[0] )


# ASYMPTOTIC LIMIT 
if(inputs['asymptotic_limits']['bool']):
	asymptotic = inputs['asymptotic_limits']
	theta0_min, theta0_max = float(asymptotic['region']['theta0_min']), float(asymptotic['region']['theta0_max'])
	theta1_min, theta1_max = float(asymptotic['region']['theta1_min']), float(asymptotic['region']['theta1_max'])
	theta2_min, theta2_max = float(asymptotic['region']['theta2_min']), float(asymptotic['region']['theta2_max'])
	resolution = int(asymptotic['region']['resolution'])
	resolutions = [resolution,resolution,1]
	n_samples_theta = int(asymptotic['n_samples_per_theta'])
	xsec = asymptotic['include_xsec']

	limits = AsymptoticLimits(h5_file)
	theta_true = [ float(asymptotic['theta_true'][0]),\
				   float(asymptotic['theta_true'][1]), \
				   float(asymptotic['theta_true'][2]) ]


	# rate + save
	theta_grid , p_values_expected_xsec, best_fit_expected_xsec = limits.expected_limits(
	theta_true=theta_true,
	theta_ranges=[(theta0_min, theta0_max), (theta1_min, theta1_max), (theta2_min, theta2_max)],
	mode="rate",
	include_xsec=True,
	resolutions=resolutions,
	luminosity=uselumi*1000.)
	
	resultsdir = '/home/results/'+method+'/rate'
	os.makedirs(resultsdir)
	np.save(resultsdir+'/grid.npy',theta_grid)
	np.save(resultsdir+'/rate.npy',[p_values_expected_xsec, best_fit_expected_xsec])
	np.save('/home/rates/rate.npy',[p_values_expected_xsec, best_fit_expected_xsec])
    
    
    #rates
	sampler_rates = SampleAugmenter(h5_file, include_nuisance_parameters=False)
	xs_grid=[]
	neff_grid=[]
	ntot=10000

	for theta_element in theta_grid:
		_,xs,_=sampler_rates.cross_sections(theta=sampling.morphing_point(theta_element))
		_,_,neff=sampler_rates.sample_train_plain(theta=sampling.morphing_point(theta_element),n_samples=ntot)
		xs_grid.append(xs)
		neff_grid.append(neff/float(ntot))
	neff_grid=np.array(neff_grid)
	xsgrid=np.array(xs_grid)

	np.save('/home/rates/neff_grid.npy',neff_grid)
	np.save('/home/rates/xs_grid.npy',xs_grid)

	#plot the rates
	xs_grid=np.load('/home/rates/xs_grid.npy')
	neff_grid=np.load('/home/rates/neff_grid.npy')
	[p_values_expected_rate, best_fit_expected_rate]= np.load('/home/rates/rate.npy')

	side_x = np.linspace(theta0_min, theta0_max, resolution)
	side_y = np.linspace(theta1_min, theta1_max, resolution)
	X1, X2 = np.meshgrid(side_x, side_y)
	size = X1.shape
	x1_1d = X1.reshape((1, np.prod(size)))
	x2_1d = X2.reshape((1, np.prod(size)))

	xdata = np.vstack((x1_1d, x2_1d))

	ydata=[ y[0] for y in xs_grid]
	initial_guess = (10**-7,10**-7,10**-7,10**-7,10**-7,10**-7)
	popt, _ = curve_fit(func, xdata, ydata=ydata, p0=initial_guess)

	newresolution=resolution

	new_grid = [ [x,y] for y in  np.linspace(theta0_min, theta0_max, newresolution)  for x in  np.linspace(theta1_min, theta1_max, newresolution)  ] 

	fit_grid = [ [func([x,y], popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])] for y in  np.linspace(theta0_min, theta0_max, newresolution)  for x in  np.linspace(theta1_min, theta1_max, newresolution)  ]
	fit_grid=np.array(fit_grid)

	rel_grid = [ [funcrel([x,y], popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])] for y in  np.linspace(theta0_min, theta0_max, newresolution)  for x in  np.linspace(theta1_min, theta1_max, newresolution)  ]
	rel_grid=np.array(rel_grid)


	#get bin sizes and positions
	theta0_bin_size = (theta0_max - theta0_min)/(resolution - 1)
	theta0_edges = np.linspace(theta0_min - theta0_bin_size/2, theta0_max + theta0_bin_size/2, resolution + 1)
	theta0_centers = np.linspace(theta0_min, theta0_max, resolution)

	theta1_bin_size = (theta1_max - theta1_min)/(resolution - 1)
	theta1_edges = np.linspace(theta1_min - theta1_bin_size/2, theta1_max + theta1_bin_size/2, resolution + 1)
	theta1_centers = np.linspace(theta1_min, theta1_max, resolution)

	#define plot
	fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
	fig.set_size_inches(12,10)

	#################
	# Panel 1: XS
	#################

	#xsec grid
	cmin1 = min(xs_grid)
	cmax1 = max(xs_grid)
	pcm1 = ax1.pcolormesh(
	    theta0_edges, theta1_edges, xs_grid.reshape((resolution, resolution)),
	    norm=matplotlib.colors.LogNorm(vmin=cmin1, vmax=cmax1),
	    cmap='Greys_r'
	)
	cbar1 = fig.colorbar(pcm1, ax=ax1, extend='both')
	cbar1.set_label('Cross Section')

	#contours for different methods
	ax1.contour(
	    theta0_centers, theta0_centers, p_values_expected_rate.reshape((resolution, resolution)),
	    levels=[0.61],
	    linestyles='-', colors='darkgreen'
	)

	# Best fit point
	ax1.scatter(
	    theta_grid[best_fit_expected_rate][0], theta_grid[best_fit_expected_rate][1],
	    s=80., color='darkgreen', marker='*',
	    label="xsec"
	)

	#Finish plot
	ax1.legend()
	ax1.set_xlabel(r'$\theta_0$')
	ax1.set_ylabel(r'$\theta_1$')
	
	#################
	# Panel 2: P-value
	#################

	#xsec grid
	cmin2, cmax2 = 1.e-3, 1.
	pcm2 = ax2.pcolormesh(
	    theta0_edges, theta1_edges, p_values_expected_rate.reshape((resolution, resolution)),
	    norm=matplotlib.colors.LogNorm(vmin=cmin2, vmax=cmax2),
	    cmap='Greys_r'
	)
	cbar2 = fig.colorbar(pcm2, ax=ax2, extend='both')
	cbar2.set_label('Expected p-value (Rate Only)')

	#contours for different methods
	ax2.contour(
	    theta0_centers, theta0_centers, p_values_expected_rate.reshape((resolution, resolution)),
	    levels=[0.61],
	    linestyles='-', colors='darkgreen'
	)

	# Best fit point
	ax2.scatter(
	    theta_grid[best_fit_expected_rate][0], theta_grid[best_fit_expected_rate][1],
	    s=80., color='darkgreen', marker='*',
	    label="xsec"
	)

	#Finish plot
	ax2.legend()
	ax2.set_xlabel(r'$\theta_0$')
	ax2.set_ylabel(r'$\theta_1$')
	      
	      
	#################
	# Panel 3: Dim6 squared 
	#################

	#xsec grid
	cmin3 = 0.1
	cmax3 = 10
	pcm3 = ax3.pcolormesh(
	    theta0_edges, theta1_edges, rel_grid.reshape((newresolution, newresolution)),
	    norm=matplotlib.colors.LogNorm(vmin=cmin3, vmax=cmax3),
	    cmap='Greys_r'
	)
	cbar3 = fig.colorbar(pcm3, ax=ax3, extend='both')
	cbar3.set_label(r"$\sigma_{dim8}/(\sigma_{tot})$")

	#contours for different methods
	ax3.contour(
	    theta0_centers, theta0_centers, rel_grid.reshape((newresolution, newresolution)),
	    levels=[0.1],
	    linestyles='-', colors='red'
	)

	# Best fit point
	ax3.scatter(
	    0, 0,
	    s=80., color='red', marker='*',
	)

	#Finish plot
	#ax3.legend()
	ax3.set_xlabel(r'$\theta_0$')
	ax3.set_ylabel(r'$\theta_1$')
	      
	#################
	# Panel 4: neffective
	#################

	#xsec grid
	cmin4, cmax4 = 1.e-2, 1.
	pcm4 = ax4.pcolormesh(
	    theta0_edges, theta1_edges, neff_grid.reshape((resolution, resolution)),
	    norm=matplotlib.colors.LogNorm(vmin=cmin4, vmax=cmax4),
	    cmap='Greys_r'
	)
	cbar4 = fig.colorbar(pcm4, ax=ax4, extend='both')
	cbar4.set_label('Effective Fraction of Events ')

	#Finish plot
	#ax4.legend()
	ax4.set_xlabel(r'$\theta_0$')
	ax4.set_ylabel(r'$\theta_1$')

	#################
	# Finish
	#################

	plt.tight_layout()
	plt.savefig('/home/rates/rates.png')


    #################### DISTRIBUTIONS

	thetas0=[np.array([x,0,0]) for x in np.arange(-2,2.1,1)]
	labels0=[r'$\theta_0=$'+str(x)  for x in np.arange(-2,2.1,1)]
	thetas1=[np.array([0,x,0]) for x in np.arange(-2,2.1,1)]
	labels1=[r'$\theta_1=$'+str(x)  for x in np.arange(-2,2.1,1)]

	fig = plot_distributions(
		filename=h5_file,
		parameter_points=thetas0,
		line_labels=labels0,
		observables=[ unicode(str(asymptotic['hist_vars'])) ],
		n_bins=20,               
		normalize=True, 
		uncertainties="none"
	)

	fig = plot_distributions(
		filename=h5_file,
		parameter_points=thetas1,
		line_labels=labels1,
		observables=[ unicode(str(asymptotic['hist_vars'])) ],
		n_bins=20,               
		normalize=True, 
		uncertainties="none"
	)






	for bool_xsec in xsec:

		# histogram + save
		_ , p_values_expected_histo, best_fit_expected_histo = limits.expected_limits(
		theta_true=theta_true,
		theta_ranges=[(theta0_min, theta0_max), (theta1_min, theta1_max), (theta2_min, theta2_max)],
		mode="histo",
		hist_vars=[ unicode(str(asymptotic['hist_vars'])) ],
		include_xsec=bool_xsec,
		resolutions=resolutions,
		luminosity=uselumi*1000.)

		resultsdir = '/home/results/'+method+'/histo'
		if not os.path.isdir(resultsdir):
			os.makedirs(resultsdir)
		if (bool_xsec==True):
			np.save(resultsdir+'/histo.npy',[p_values_expected_histo, best_fit_expected_histo])
		else:
			np.save(resultsdir+'/histo_kin.npy',[p_values_expected_histo, best_fit_expected_histo])

		# method ML +save
		if( method in ['alice','alices','cascal','carl','rolr', 'rascal'] ):
			theta_grid, p_values_expected_method, best_fit_expected_method = limits.expected_limits(
			theta_true=theta_true,
			theta_ranges=[(theta0_min, theta0_max), (theta1_min, theta1_max), (theta2_min, theta2_max)],
			mode="ml",
			model_file=eval_folder_path+'/'+method,
			include_xsec=bool_xsec,
			resolutions=resolutions,
			luminosity=uselumi*1000.)
			resultsdir = '/home/results/'+method+'/ml'
			if not os.path.isdir(resultsdir):
				os.makedirs(resultsdir)
			if (bool_xsec==True):
				np.save(resultsdir+'/'+method+'.npy',[p_values_expected_method, best_fit_expected_method])
			else:
				np.save(resultsdir+'/'+method+'_kin.npy',[p_values_expected_method, best_fit_expected_method])

			

		#histo method + save
		if( method in ['sally', 'sallino'] ):
			theta_grid , p_values_expected_method, best_fit_expected_method = limits.expected_limits(
			theta_true=theta_true,
			theta_ranges=[(theta0_min, theta0_max), (theta1_min, theta1_max), (theta2_min, theta2_max)],
			mode="histo",
			model_file= eval_folder_path+'/'+method, 
			include_xsec=bool_xsec,
			resolutions=resolutions,
			luminosity=uselumi*1000.)

			resultsdir = '/home/results/'+method+'/histo'
			if not os.path.isdir(resultsdir):
				os.makedirs(resultsdir)
			if (bool_xsec==True):
				np.save(resultsdir+'/'+method+'.npy',[p_values_expected_method, best_fit_expected_method])
			else:
				np.save(resultsdir+'/'+method+'_kin.npy',[p_values_expected_method, best_fit_expected_method])
			


# FISHER INFO
if( bool(inputs['fisher_information']['bool']) and (method in ['sally']) ):

  uselumi=300*1000.
  fisher_input = inputs['fisher_information']
  fisher = FisherInformation(h5_file, include_nuisance_parameters=False)
  theta_true = [ float(fisher_input['theta_true'][0]),\
				   float(fisher_input['theta_true'][1]), \
				   float(fisher_input['theta_true'][2]) ]

  fi_rate, fi_rate_cov = fisher.calculate_fisher_information_rate(
    theta=theta_true,
    luminosity=uselumi*1000.,
  )

  fi_histo, fi_histo_cov = fisher.calculate_fisher_information_hist1d(
    theta=[0.,0.,0.],
    luminosity=theta_true,
    observable=str(fisher_input['observable']), 
    nbins=10,
    histrange=(0,1000)
  )


  fi_sally, fi_sally_cov = fisher.calculate_fisher_information_full_detector(
    theta=theta_true,
    model_file=eval_folder_path+'/models/sally',
    unweighted_x_sample_file=eval_folder_path+'/test/x_test.npy',
    luminosity=uselumi*1000.,
    include_xsec_info=True,
  )  

  
  fi2d_rate, fi2d_rate_cov = project_information(fisher_information=fi_rate,
    covariance=fi_rate_cov,remaining_components=[0,1])
  fi2d_histo, fi2d_histo_cov = project_information(fisher_information=fi_histo,
    covariance=fi_histo_cov,remaining_components=[0,1])
  fi2d_sally, fi2d_sally_cov = project_information(fisher_information=fi_sally,
    covariance=fi_sally_cov,remaining_components=[0,1])
  
  fi2d_histo_kin = fi2d_histo - fi2d_rate
  fi2d_sally_kin = fi2d_sally - fi2d_rate

  list_fi = [fi2d_rate, fi2d_histo,fi2d_histo_kin, fi2d_sally,fi2d_sally_kin]
  list_fi_cov = [fi2d_rate_cov, fi2d_histo_cov,None, fi2d_sally_cov,None]
  list_fi_labels = ['Rate','Histo','Histo-Kin','SALLY','SALLY-Kin']
  list_fi_colors= ['black','red','orange','darkgreen','limegreen']

  contourplot = plot_fisher_information_contours_2d(
    fisher_information_matrices=list_fi, 
    fisher_information_covariances=list_fi_cov,
    inline_labels=list_fi_labels,
    contour_distance=1,
    xlabel=r'$\theta_0$',
    ylabel=r'$\theta_1$',
    xrange=(-0.5,0.5),
    yrange=(-0.5,0.5),
    resolution=100
  )
		
  contourplot.savefig('/home/plots/plot_fisher.png')




# # EVALUATE: TRAIN VS TEST
# forge = Estimator() 
# forge.load(eval_folder_path+'/'+method)  #'methods/alices'


# #?
# theta_denom = np.array([[0.,0.]])
# np.save('/home/data/test/theta_ref.npy', theta_denom)

# print('you need to update v0.3.0 to forge.evaluate_likelihood() or similar ')


# #perform the test + evaluation score acconding to method
# if( method  in ['alice',  'alices',  'carl',  'nde', 'rascal',  'rolr',  'scandal'] ):
	
# 	#make and save grid
# 	evaluation = inputs['evaluation']['theta_each']
# 	theta_each = np.linspace( float(evaluation['start']), float(evaluation['stop']), int(evaluation['num']) ) 
# 	theta0, theta1 = np.meshgrid(theta_each, theta_each)
# 	theta_grid = np.vstack((theta0.flatten(), theta1.flatten())).T #numtidim
# 	np.save('/home/data/test/theta_grid.npy', theta_grid)

# 	log_r_hat, score_theta0, _ = forge.evaluate(
# 	    theta0_filename='/home/data/test/theta_grid.npy',
# 	    x='/home/data/test/x_test.npy',
# 	    evaluate_score=inputs['evaluation']['evaluate_score']
# 	)
# 	with open('/home/data/test/log_r_hat_'+method+'.npy', "w+") as f: #create file
# 		np.save(file='/home/data/test/log_r_hat_'+method, arr=log_r_hat)
	
# 	with open('/home/data/test/score_theta0_'+method+'.npy', "w+") as g: #create file
# 		np.save(file='/home/data/test/score_theta0_'+method, arr=score_theta0)
	

# 	#plots
# 	if( bool(inputs['plots']['activate']) ):
		
# 		bin_size = theta_each[1] - theta_each[0]
# 		edges = np.linspace(theta_each[0] - bin_size/2, theta_each[-1] + bin_size/2, len(theta_each)+1)

# 		fig = plt.figure(figsize=(6,5))
# 		ax = plt.gca()

# 		expected_llr = np.mean(log_r_hat,axis=1)
# 		best_fit = theta_grid[np.argmin(-2.*expected_llr)]

# 		cmin, cmax = np.min(-2*expected_llr), np.max(-2*expected_llr)
			
# 		pcm = ax.pcolormesh(edges, edges, -2. * expected_llr.reshape((21,21)),
# 		                    norm=matplotlib.colors.Normalize(vmin=cmin, vmax=cmax),
# 		                    cmap='viridis_r')
# 		cbar = fig.colorbar(pcm, ax=ax, extend='both')

# 		plt.scatter(best_fit[0], best_fit[1], s=80., color='black', marker='*')

# 		plt.xlabel(r'$\theta_0$')
# 		plt.ylabel(r'$\theta_1$')
# 		cbar.set_label(r'$\mathbb{E}_x [ -2\, \log \,\hat{r}(x | \theta, \theta_{SM}) ]$')

# 		plt.tight_layout()
# 		plt.savefig('/home/plots/expected_llr_'+method+'.png')	


# if( method  in ['alice2', 'alices2', 'carl2', 'rascal2', 'rolr2' ] ):
# 	print('evaluation for this method is not yet implemented')
# 	pass 

# 	#make and save grid
# 	evaluation = inputs['evaluation']['theta_each']
# 	theta_each = np.linspace( float(evaluation['start']), float(evaluation['stop']), int(evaluation['num']) ) 
# 	theta0, theta1 = np.meshgrid(theta_each, theta_each)
# 	theta_grid = np.vstack((theta0.flatten(), theta1.flatten())).T #numtidim
# 	np.save('/home/data/test/theta_grid.npy', theta_grid)

# 	log_r_hat, score_theta0, score_theta1 = forge.evaluate(
# 	    theta0_filename='/home/data/test/theta_grid.npy',
# 	    theta1_filename='/home/data/test/theta_grid.npy', #TO DO !
# 	    x='/home/data/test/x_test.npy',
# 	    evaluate_score=False
# 	)
# 	with open('/home/data/test/log_r_hat_'+method+'.npy', "w+") as f: #create file
# 		np.save(file='/home/data/test/log_r_hat_'+method, arr=log_r_hat)		


# if( method  in ['sally', 'sallino'] ):
# 	t_hat = forge.evaluate(
#     	x='/home/data/samples/x_test.npy'
# 	)
# 	with open('/home/data/test/t_hat_'+method+'.npy', "w+") as f: #create file
# 		np.save(file='/home/data/test/t_hat_'+method, arr=t_hat)

# 	#plots
# 	if( bool(inputs['plots']['activate']) ):
# 		x = np.load('data/samples/x_test.npy')
# 		fig = plt.figure(figsize=(10,4))

# 		for i in range(2):
# 			ax = plt.subplot(1,2,i+1)
# 			sc = plt.scatter(x[::10,0], x[::10,1], c=t_hat[::10,i], s=10., cmap='viridis', vmin=-0.8, vmax=0.4)
# 			cbar = plt.colorbar(sc)
# 			cbar.set_label(r'$\hat{t}_' + str(i) + r'(x | \theta_{ref})$')
# 			plt.xlabel(r'$p_{T,j1}$ [GeV]')
# 			plt.ylabel(r'$\Delta \phi_{jj}$')
# 			plt.xlim(10.,400.)
# 			plt.ylim(-3.15,3.15)
# 			plt.tight_layout()
# 			plt.savefig('/home/plots/t_hat_'+method+'.png')	


