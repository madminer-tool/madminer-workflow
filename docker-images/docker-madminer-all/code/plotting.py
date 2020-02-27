from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import math
import random
import sys
import os, glob, glob2
import yaml
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from madminer.limits import AsymptoticLimits
from madminer.sampling import SampleAugmenter
from madminer import sampling

logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.INFO
)

for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)


#######################
def do_plot(
    p_values_expected,
    best_fit_expected,
    colors,
    styles,
    titles,
    n_cols=3,
    theta0range=[-1,1],
    theta1range=[-1,1],
    resolution=10
    ):
    
    #get bin sizes and positions
    theta0_min=theta0range[0]
    theta0_max=theta0range[1]
    theta0_bin_size = (theta0_max - theta0_min)/(resolution - 1)
    theta0_edges = np.linspace(theta0_min - theta0_bin_size/2, theta0_max + theta0_bin_size/2, resolution + 1)
    theta0_centers = np.linspace(theta0_min, theta0_max, resolution)

    theta1_min=theta0range[0]
    theta1_max=theta0range[1]
    theta1_bin_size = (theta1_max - theta1_min)/(resolution - 1)
    theta1_edges = np.linspace(theta1_min - theta1_bin_size/2, theta1_max + theta1_bin_size/2, resolution + 1)
    theta1_centers = np.linspace(theta1_min, theta1_max, resolution)
    
    cmin, cmax = 1.e-3, 1.

    # Preparing plot
    n_methods=len(p_values_expected)+1
    n_rows = (n_methods + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(6.0 * n_cols, 5.0 * n_rows))

    for ipanel in range(len(p_values_expected)):

        # Panel
        ax = plt.subplot(n_rows, n_cols, ipanel + 1)
        
        #p-value
        pcm = ax.pcolormesh(
            theta0_edges, theta1_edges, p_values_expected[ipanel].reshape((resolution, resolution)),
            norm=matplotlib.colors.LogNorm(vmin=cmin, vmax=cmax),
            cmap='Greys_r'
        )
        cbar = fig.colorbar(pcm, ax=ax, extend='both')
        cbar.set_label('Expected p-value')

        #contours 
        plt.contour(
            theta0_centers, theta0_centers, p_values_expected[ipanel].reshape((resolution, resolution)),
            levels=[0.61],
            linestyles=styles[ipanel], 
            colors=colors[ipanel],
            label=r"$1\sigma$ contour"
        )
        
        #best fit
        plt.scatter(
            variables_to_plot['theta_grid'][best_fit_expected[ipanel]][0], variables_to_plot['theta_grid'][best_fit_expected[ipanel]][1],
            s=80., color=colors[ipanel], marker='+',
            label="Best Fit"
        )    
        
        #title
        plt.title(titles[ipanel])
        plt.xlabel(r'$\theta_0$')
        plt.ylabel(r'$\theta_1$')
        plt.legend()
        
    ax = plt.subplot(n_rows, n_cols, len(p_values_expected) + 1)  
    for ipanel in range(len(p_values_expected)):

        #contours 
        plt.contour(
            theta0_centers, theta0_centers, p_values_expected[ipanel].reshape((resolution, resolution)),
            levels=[0.61],
            linestyles=styles[ipanel], 
            colors=colors[ipanel],
        )
        
        #best fit
        if styles[ipanel]=="solid":
            marker='*'
        else:
            marker='+'
        plt.scatter(
            variables_to_plot['theta_grid'][best_fit_expected[ipanel]][0], variables_to_plot['theta_grid'][best_fit_expected[ipanel]][1],
            s=80., color=colors[ipanel], marker=marker,
            label= titles[ipanel]
        )    
        
        #title
        plt.xlabel(r'$\theta_0$')
        plt.ylabel(r'$\theta_1$')
        plt.title(titles[ipanel])
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('/home/plots/all_methods_separate.png')



inputs_file = sys.argv[1] 
with open(inputs_file) as f:
	inputs = yaml.safe_load(f)

#folder with trained files
#results_folder_path = str(sys.argv[2]) 


# get variables form input
asymptotic = inputs['asymptotic_limits']
theta_ranges=[]
for this_theta in asymptotic['region']:
  theta_min, theta_max = asymptotic['region'][str(this_theta)]
  theta_ranges.append((theta_min,theta_max))

theta0_min, theta0_max = asymptotic['region']['theta0_min_max']
theta1_min, theta1_max = asymptotic['region']['theta1_min_max']
#theta2_min, theta2_max = asymptotic['region']['theta2_min_max']
n_samples_theta = int(asymptotic['n_samples_per_theta'])
resolutions = asymptotic['resolutions']
plotting =  inputs['plotting']

variables_to_plot = dict()

# LOAD 
# gridfiles = []
# for root, dirs, files in os.walk(results_folder_path):
#   for file in files:
#     if file.endswith('grid.npy'):
#       gridfiles.append(os.path.join(root, file))
# gridpath = gridfiles[0]

variables_to_plot['theta_grid'] = np.load('/home/rates/grid.npy')  #np.load(gridpath)

# ratefiles = []
# for root, dirs, files in os.walk(results_folder_path):
#   for file in files:
#     if file.endswith('rate.npy'):
#       ratefiles.append(os.path.join(root, file))
# ratepath = ratefiles[0]

variables_to_plot['p_values_expected_rate'], variables_to_plot['best_fit_expected_rate'] = np.load('/home/rates/rate.npy')  #np.load(ratepath)


# histofiles = []
# for root, dirs, files in os.walk(results_folder_path):
#   for file in files:
#     if file.endswith('histo.npy'):
#       histofiles.append(os.path.join(root, file))
# histopath = histofiles[0]

variables_to_plot['p_values_expected_histo'], variables_to_plot['best_fit_expected_histo'] = np.load('/home/rates/histo.npy')


# histo_kinfiles = []
# for root, dirs, files in os.walk(results_folder_path):
#   for file in files:
#     if file.endswith('histo_kin.npy'):
#       histo_kinfiles.append(os.path.join(root, file))
# histo_kinpath = histo_kinfiles[0]

variables_to_plot['p_values_expected_histo_kin'], variables_to_plot['best_fit_expected_histo_kin'] = np.load('/home/rates/histo_kin.npy')



# this is for the method histrograms
methods = inputs['methods']

for method in methods:
  if(method in ['alice','alices','cascal','carl','rolr', 'rascal'] ):
    variables_to_plot['p_values_expected_'+method] = np.load('/home/results/'+method+'/ml/'+method+'.npy')[0]
    variables_to_plot['best_fit_expected_'+method] = np.load('/home/results/'+method+'/ml/'+method+'.npy')[1]
    variables_to_plot['p_values_expected_'+method+'_kin'] = np.load('/home/results/'+method+'/ml/'+method+'_kin.npy')[0]
    variables_to_plot['best_fit_expected_'+method+'_kin'] = np.load('/home/results/'+method+'/ml/'+method+'_kin.npy')[1]


  if(method in ['sally','sallino'] ):
    variables_to_plot['p_values_expected_'+method] = np.load('/home/results/'+method+'/histo/'+method+'.npy')[0]
    variables_to_plot['best_fit_expected_'+method] = np.load('/home/results/'+method+'/histo/'+method+'.npy')[1]
    variables_to_plot['p_values_expected_'+method+'_kin'] = np.load('/home/results/'+method+'/histo/'+method+'_kin.npy')[0]
    variables_to_plot['best_fit_expected_'+method+'_kin'] = np.load('/home/results/'+method+'/histo/'+method+'_kin.npy')[1]    


# for method in methods:
#   methodfiles = []
#   for root, dirs, files in os.walk(results_folder_path+'/'+method):
#     for file in files:
#       if (file.endswith('.npy') and file.startswith(method)):
#         methodfiles.append(os.path.join(root, file))
  
#   #kin / not kin
#   for option_file in methodfiles: 
#     if(option_file.endswith('kin.npy')):
#       variables_to_plot['p_values_expected_'+method+'_kin'] = np.load(option_file)[0]
#       variables_to_plot['best_fit_expected_'+method+'_kin'] = np.load(option_file)[1]
#     else:
#       variables_to_plot['p_values_expected_'+method+''] = np.load(option_file)[0]
#       variables_to_plot['best_fit_expected_'+method+''] = np.load(option_file)[1]




# PLOT IT ALL 

if( plotting['all_methods']==True ):
  # get bin sizes and positions
  theta0_bin_size = (theta0_max - theta0_min)/(resolutions[0] - 1)
  theta0_edges = np.linspace(theta0_min - theta0_bin_size/2, theta0_max + theta0_bin_size/2, resolutions[0] + 1)
  theta0_centers = np.linspace(theta0_min, theta0_max, resolutions[0])
  theta1_bin_size = (theta1_max - theta1_min)/(resolutions[1] - 1)
  theta1_edges = np.linspace(theta1_min - theta1_bin_size/2, theta1_max + theta1_bin_size/2, resolutions[1] + 1)
  theta1_centers = np.linspace(theta1_min, theta1_max, resolutions[1])


  #define plot
  fig = plt.figure(figsize=(6,5))
  ax = plt.gca()

  #alices depeent
  cmin, cmax = 1.e-3, 1.
  method_pvalue = str(inputs['plotting']['all_methods_pvalue'])

  print('edges shapes.....', theta0_edges.shape, theta1_edges.shape, variables_to_plot['p_values_expected_'+method_pvalue+'_kin'].shape)

  pcm = ax.pcolormesh(
      theta0_edges, theta1_edges, variables_to_plot['p_values_expected_'+method_pvalue+'_kin'].reshape([ resolutions[0], resolutions[1] ]),
      norm=matplotlib.colors.LogNorm(vmin=cmin, vmax=cmax),
      cmap='Greys_r'
  )
  cbar = fig.colorbar(pcm, ax=ax, extend='both')


  #methods
  random.seed(5)
  for method in methods:
    print('plotting...... ',method)
    if(method=='sally'):
    	color='C0'
    if(method=='alice'):
    	color='C1'
    if(method=='alices'):
      color='red'
    if(method=='rascal'):
      color='magenta'

    plt.contour(
      theta0_centers, theta0_centers, variables_to_plot['p_values_expected_'+method].reshape([ resolutions[0], resolutions[1] ]),
      levels=[0.61],
      linestyles='-', colors=color
    )
    plt.contour(
      theta0_centers, theta0_centers, variables_to_plot['p_values_expected_'+method+'_kin'].reshape([ resolutions[0], resolutions[1] ]),
      levels=[0.61],
      linestyles='--', colors=color
    )
    plt.scatter(
      variables_to_plot['theta_grid'][variables_to_plot['best_fit_expected_'+method] ][0], variables_to_plot['theta_grid'][variables_to_plot['best_fit_expected_'+method] ][1],
      s=80., color=color, marker='*',
      label = method.upper()
    )
    plt.scatter(
      variables_to_plot['theta_grid'][variables_to_plot['best_fit_expected_'+method+'_kin'] ][0], variables_to_plot['theta_grid'][variables_to_plot['best_fit_expected_'+method+'_kin'] ][1],
      s=80., color=color, marker='+',
      label = (method+'-kin').upper()
    )


  #rate
  plt.contour(
    theta0_centers, theta0_centers, variables_to_plot['p_values_expected_rate'].reshape([ resolutions[0], resolutions[1] ]),
    levels=[0.61],
    linestyles='-', colors='black'
    
  )
  plt.scatter(
    variables_to_plot['theta_grid'][variables_to_plot['best_fit_expected_rate'] ][0], variables_to_plot['theta_grid'][variables_to_plot['best_fit_expected_rate'] ][1],
    s=80., color='black', marker='*',
    label="xsec"
    )

  #histo
  plt.contour(
    theta0_centers, theta0_centers, variables_to_plot['p_values_expected_histo'].reshape([ resolutions[0], resolutions[1] ]),
    levels=[0.61],
    linestyles='-', colors='limegreen',
    label="Histo"
  )

  plt.scatter(
    variables_to_plot['theta_grid'][variables_to_plot['best_fit_expected_histo'] ][0], variables_to_plot['theta_grid'][variables_to_plot['best_fit_expected_histo'] ][1],
    s=80., color='limegreen', marker='*',
    label="Histo"
  )

  #kin
  plt.contour(
    theta0_centers, theta0_centers, variables_to_plot['p_values_expected_histo_kin'].reshape([ resolutions[0], resolutions[1] ]),
    levels=[0.61],
    linestyles='--', colors='limegreen'
  )


  plt.scatter(
    variables_to_plot['theta_grid'][variables_to_plot['best_fit_expected_histo_kin'] ][0], variables_to_plot['theta_grid'][variables_to_plot['best_fit_expected_histo_kin'] ][1],
    s=80., color='limegreen', marker='+',
    label="Histo-Kin"
    )


  #Finish plot
  plt.legend()
  plt.xlabel(r'$\theta_0$')
  plt.ylabel(r'$\theta_1$')
  cbar.set_label('Expected p-value ('+method_pvalue.upper()+'-Kinematics)')

  plt.tight_layout()
  plt.savefig('/home/plots/all_methods.png')




if( plotting['all_methods_separate'] == True ):

  plotinput=[
    [variables_to_plot['p_values_expected_rate']      ,variables_to_plot['best_fit_expected_rate']      ,"Rate"      ,"black"    ,"solid" ],
    [variables_to_plot['p_values_expected_histo']     ,variables_to_plot['best_fit_expected_histo']     ,"Histo"     ,"limegreen","solid" ],
    [variables_to_plot['p_values_expected_histo_kin'] ,variables_to_plot['best_fit_expected_histo_kin'] ,"Histo-Kin" ,"limegreen","dashed"],
    [variables_to_plot['p_values_expected_sally']     ,variables_to_plot['best_fit_expected_sally']     ,"SALLY"     ,"C0"       ,"solid" ],
    [variables_to_plot['p_values_expected_sally_kin'] ,variables_to_plot['best_fit_expected_sally_kin'] ,"SALLY-Kin" ,"C0"       ,"dashed"],
    [variables_to_plot['p_values_expected_alices']    ,variables_to_plot['best_fit_expected_alices']    ,"ALICES"    ,"red"      ,"solid" ],
    [variables_to_plot['p_values_expected_alices_kin'],variables_to_plot['best_fit_expected_alices_kin'],"ALICES-Kin","red"      ,"dashed"]
  ]

  plotinput=np.array(plotinput)

  do_plot(
    p_values_expected=plotinput[:,0],
    best_fit_expected=plotinput[:,1],
    colors=plotinput[:,3],
    styles=plotinput[:,4],
    titles=plotinput[:,2],
    theta0range=[theta0_min,theta0_max],
    theta1range=[theta1_min,theta1_max],
    resolution=resolutions[0]
  )


if(plotting['correlations']==True):

  for method in inputs['plotting']['correlations_methods']:


    if(method in ['alice','alices','cascal','carl','rolr', 'rascal']):
      # Joint LLR
      if(len(inputs['evaluation'][method])==1):
        
        theta_test=np.load('/home/test/'+method+'/theta_test.npy')

        r_truth_test=np.load('/home/test/'+method+'/r_xz_test.npy')
        r_truth_test=r_truth_test.flatten()
        llr_truth_test=[math.log(r) for r in r_truth_test ] 
        llr_truth_test=np.array(llr_truth_test)

        # Estimated LLR
        from madminer.ml import ParameterizedRatioEstimator
        estimator_load = ParameterizedRatioEstimator()
        estimator_load.load('/home/models/'+method+'/'+method)

        llr_ml_test,_=estimator_load.evaluate_log_likelihood_ratio(
          theta='/home/test/'+method+'/theta_test.npy',
          x='/home/test/'+method+'/x_test.npy',
          evaluate_score=False,
          test_all_combinations=False,
        )

        #Define Colormap
        from matplotlib import cm
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap
        oldcolors = cm.get_cmap('Greens', 256)
        newcolors = oldcolors(np.linspace(0, 1, 256))
        newcolors[:5, :] = np.array([1, 1, 1, 1])
        newcolormap = ListedColormap(newcolors)


        #Plot
        myrange=(-2,2)

        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(5,5)

        ax.hist2d(
          llr_truth_test,
          llr_ml_test,
          bins=(40, 40), 
          range=(myrange,myrange),
          cmap=newcolormap,
        )

        ax.set_xlabel('LLR: Truth')
        ax.set_ylabel('LLR: Estimated')

        plt.title('Correlation '+method.upper())
        plt.tight_layout()
        plt.savefig('/home/plots/correlation_'+method.upper()+'.png')



      if(len(inputs['evaluation'][method])==2):
        theta0_test=np.load('/home/test/'+method+'/theta0_test.npy')
        theta1_test=np.load('/home/test/'+method+'/theta1_test.npy')

        r_truth_test=np.load('/home/test/'+method+'/r_xz_test.npy')
        r_truth_test=r_truth_test.flatten()
        llr_truth_test=[math.log(r) for r in r_truth_test ] 
        llr_truth_test=np.array(llr_truth_test)

        # Estimated LLR
        from madminer.ml import ParameterizedRatioEstimator
        estimator_load = ParameterizedRatioEstimator()
        estimator_load.load('/home/models/'+method+'/'+method)

        llr_ml_test,_=estimator_load.evaluate_log_likelihood_ratio(
          theta='/home/test/'+method+'/theta0_test.npy',
          x='/home/test/'+method+'/x_test.npy',
          evaluate_score=False,
          test_all_combinations=False,
        )

        #Define Colormap
        from matplotlib import cm
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap
        oldcolors = cm.get_cmap('Greens', 256)
        newcolors = oldcolors(np.linspace(0, 1, 256))
        newcolors[:5, :] = np.array([1, 1, 1, 1])
        newcolormap = ListedColormap(newcolors)


        #Plot
        myrange=(-2,2)

        fig, ax = plt.subplots(1,1)
        fig.set_size_inches(5,5)

        ax.hist2d(
          llr_truth_test,
          llr_ml_test,
          bins=(40, 40), 
          range=(myrange,myrange),
          cmap=newcolormap,
        )

        ax.set_xlabel('LLR: Truth')
        ax.set_ylabel('LLR: Estimated')

        plt.title('Correlation '+method.upper())
        plt.tight_layout()
        plt.savefig('/home/plots/correlation_'+method.upper()+'.png')

    

    if(method in ['sally','sallino']):

      pass
      
      # Joint LLR
      theta0_test=np.load('/home/test/'+method+'/theta0_test.npy')

      r_truth_test=np.load('/home/test/'+method+'/r_xz_test.npy')
      r_truth_test=r_truth_test.flatten()
      llr_truth_test=[math.log(r) for r in r_truth_test ] 
      llr_truth_test=np.array(llr_truth_test)

      # Estimated LLR
      from madminer.ml import ParameterizedRatioEstimator
      estimator_load = ParameterizedRatioEstimator()
      estimator_load.load('/home/models/'+method+'/'+method)

      llr_ml_test,_=estimator_load.evaluate_log_likelihood_ratio(
        theta='/home/test/'+method+'/theta0_test.npy',
        x='/home/test/'+method+'/x_test.npy',
        evaluate_score=False,
        test_all_combinations=False,
      )

      #Define Colormap
      from matplotlib import cm
      from matplotlib.colors import ListedColormap, LinearSegmentedColormap
      oldcolors = cm.get_cmap('Greens', 256)
      newcolors = oldcolors(np.linspace(0, 1, 256))
      newcolors[:5, :] = np.array([1, 1, 1, 1])
      newcolormap = ListedColormap(newcolors)


      #Plot
      myrange=(-2,2)

      fig, ax = plt.subplots(1,1)
      fig.set_size_inches(5,5)

      ax.hist2d(
        llr_truth_test,
        llr_ml_test,
        bins=(40, 40), 
        range=(myrange,myrange),
        cmap=newcolormap,
      )

      ax.set_xlabel('LLR: Truth')
      ax.set_ylabel('LLR: Estimated')

      plt.title('Correlation '+method.upper())
      plt.tight_layout()
      plt.savefig('/home/plots/correlation_'+method.upper()+'.png')



