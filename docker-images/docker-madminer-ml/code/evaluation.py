from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import os
import logging
import numpy as np
import math
import matplotlib
import glob
import yaml
import h5py


from madminer.limits import AsymptoticLimits
from madminer import sampling
from madminer.sampling import SampleAugmenter
from madminer.ml import ParameterizedRatioEstimator, ScoreEstimator, Ensemble, DoubleParameterizedRatioEstimator, LikelihoodEstimator
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from madminer.fisherinformation import FisherInformation
from madminer.fisherinformation import project_information,profile_information

from madminer.plotting import plot_fisher_information_contours_2d
from madminer.plotting import plot_distributions

from madminer.sampling import benchmark, benchmarks, random_morphing_points, morphing_point

from madminer.analysis import DataAnalyzer
from madminer.utils.interfaces.madminer_hdf5 import madminer_event_loader
from madminer.utils.interfaces.madminer_hdf5 import save_preformatted_events_to_madminer_file
from madminer.utils.various import create_missing_folders, shuffle



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

def generate_test_data_ratio(method):
    # get number of paramenters
    hf = h5py.File(h5_file, 'r')
    parameters = len(hf['parameters']['names'])
    sa = SampleAugmenter(h5_file, include_nuisance_parameters=False)


    if( len(inputs['evaluation'][str(method)])==1 ): #only one theta

      theta_sampling = inputs['evaluation'][str(method)]['theta']['sampling_method']
      theta = inputs['evaluation'][str(method)]['theta']
      if(theta_sampling != 'random_morphing_points'):

        x, theta, y, r_xz, t_xz, n_effective = sa.sample_test(
        theta=eval(theta_sampling)(theta['argument']),
        n_samples=inputs['n_samples']['test'],
        folder='/madminer/test/'+method+'/',
        filename='test',
        switch_train_test_events=True
        )

      else:

        prior = []
        for p in range(parameters):
            this_tuple = theta['prior']['parameter_'+str(p)]
            prior.append( (str(this_tuple['prior_shape']), float(this_tuple['prior_param_0']), float(this_tuple['prior_param_1'])) )

        x, theta, y, r_xz, t_xz, n_effective = sa.sample_test(
        theta=eval(theta_sampling)(theta_['n_thetas'], prior),
        n_samples=inputs['n_samples']['test'],
        folder='/madminer/test/'+method+'/',
        filename='test',
        switch_train_test_events=True,
        )


    elif( len(inputs['evaluation'][str(method)])==2 ): #two thetas

      theta0_sampling = inputs['evaluation'][str(method)]['theta_0']['sampling_method'] #sampling method for theta0
      theta1_sampling = inputs['evaluation'][str(method)]['theta_1']['sampling_method'] #sampling method for theta1
      theta_0 = inputs['evaluation'][str(method)]['theta_0'] #parameters for theta0 sampling
      theta_1 = inputs['evaluation'][str(method)]['theta_1'] #parameters for theta0 sampling

      if (theta0_sampling == 'random_morphing_points' and theta1_sampling != 'random_morphing_points' ): 
          
          prior = []
          for p in range(parameters):
              this_tuple = theta_0['prior']['parameter_'+str(p)]
              prior.append( (str(this_tuple['prior_shape']), float(this_tuple['prior_param_0']), float(this_tuple['prior_param_1'])) )


          x,th0,th1,y,r_xz,t_xz = sa.sample_train_ratio(
              theta0=eval(theta0_sampling)(theta_0['n_thetas'], prior),
              theta1=eval(theta1_sampling)(theta_1['argument']),
              n_samples=inputs['n_samples']['test'],
              folder='/madminer/test/'+method+'/',
              filename='test',
              switch_train_test_events=True,
              )      
              

      elif (theta1_sampling == 'random_morphing_points' and theta0_sampling != 'random_morphing_points'):  
          tuple_0 = theta_1['prior']['parameter_0'] #tuple for parameter 0
          tuple_1 = theta_1['prior']['parameter_1'] #tuple for parameter 1
          prior = [ (str(tuple_0['prior_shape']), float(tuple_0['prior_param_0']), float(tuple_0['prior_param_1'])), \
                    (str(tuple_1['prior_shape']), float(tuple_1['prior_param_0']), float(tuple_1['prior_param_1']))  ]

          x, theta0, theta1, y, r_xz, t_xz = sa.sample_train_ratio(
              theta0=eval(theta0_sampling)(theta_0['argument']),
              theta1=eval(theta1_sampling)(theta_1['n_thetas'], prior),
              n_samples=inputs['n_samples']['test'],
              folder='/madminer/test/'+method+'/',
              filename='test',
              switch_train_test_events=True,     
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

          x, theta0, theta1, y, r_xz, t_xz = sa.sample_train_ratio(
              theta0=eval(theta0_sampling)(theta_0['n_thetas'], prior0),
              theta1=eval(theta1_sampling)(theta_1['n_thetas'], prior1),
              n_samples=inputs['n_samples']['test'],
              folder='/madminer/test/'+method+'/',
              filename='test',
              switch_train_test_events=True,     
          )


      else:
          x, theta0, theta1, y, r_xz, t_xz, n_effective= sa.sample_train_ratio(
              theta0=eval(theta0_sampling)(theta_0['argument']),
              theta1=eval(theta1_sampling)(theta_1['argument']),
              n_samples=inputs['n_samples']['test'],
              folder='/madminer/test/'+method+'/',
              filename='test',
              switch_train_test_events=True
          )

def generate_test_data_score(method):
    # get number of paramenters
    hf = h5py.File(h5_file, 'r')
    parameters = len(hf['parameters']['names'])
    sa = SampleAugmenter(h5_file, include_nuisance_parameters=False)

    theta_input = inputs[str(method)]['theta']
    theta_sampling = theta_input['sampling_method']
    
    if (theta_sampling == 'random_morphing_points'): 
                
        prior = []
        for p in range(parameters):
            this_tuple = theta_input['prior']['parameter_'+str(p)]
            prior.append( (str(this_tuple['prior_shape']), float(this_tuple['prior_param_0']), float(this_tuple['prior_param_1'])) )


        x, theta0, theta1, y, r_xz, t_xz = sample_train_local(
            theta=eval(theta_sampling)(theta_input['n_thetas'], prior),
            n_samples=inputs['n_samples']['test'],
            folder='/madminer/test/'+method+'/',
            filename='test',
            switch_train_test_events=False,      
        )

    if (theta_sampling == 'benchmark'): 
        _ = sa.sample_train_local(
            theta=eval(theta_sampling)(theta_input['argument']),
            n_samples=inputs['n_samples']['test'],
            folder='/madminer/test/'+method+'/',
            filename='test',
            switch_train_test_events=False,
        )



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

path_split = os.path.split(os.path.abspath(eval_folder_path))

method = str(path_split[1]) 



# ASYMPTOTIC LIMIT 
if(inputs['asymptotic_limits']['bool']):

    asymptotic = inputs['asymptotic_limits']

    theta_ranges=[]
    for this_theta in asymptotic['region']:
      theta_min, theta_max = asymptotic['region'][str(this_theta)]
      theta_ranges.append((theta_min,theta_max))
    
    print('theta range...', theta_ranges)
    
    resolutions = asymptotic['resolutions']
    print('resolutions...',resolutions)
    n_samples_theta = int(asymptotic['n_samples_per_theta'])
    xsec = asymptotic['include_xsec']
    theta_true = asymptotic['theta_true']


    limits = AsymptoticLimits(h5_file)
    
    #################### RATES & GRID

    theta_grid, p_values_expected_xsec, best_fit_expected_xsec = limits.expected_limits(
    theta_true=theta_true,
    theta_ranges=theta_ranges,
    mode="rate",
    include_xsec=True,
    resolutions=resolutions,
    luminosity=uselumi)
    
    np.save('/madminer/rates/grid.npy',theta_grid)
    np.save('/madminer/rates/rate.npy',[p_values_expected_xsec, best_fit_expected_xsec])
    
    
    sa_rates = SampleAugmenter(h5_file, include_nuisance_parameters=False)
    xs_grid=[]
    neff_grid=[]
    n_test=10000

    for theta_element in theta_grid:
        _,xs,_=sa_rates.cross_sections(theta=sampling.morphing_point(theta_element))
        _,_,neff=sa_rates.sample_train_plain(theta=sampling.morphing_point(theta_element),n_samples=n_test)
        xs_grid.append(xs)
        neff_grid.append(neff/float(n_test))
    neff_grid=np.array(neff_grid)
    xsgrid=np.array(xs_grid)

    np.save('/madminer/rates/neff_grid.npy',neff_grid)
    np.save('/madminer/rates/xs_grid.npy',xs_grid)


    for bool_xsec in xsec:
        # histogram + save
        _ , p_values_expected_histo, best_fit_expected_histo = limits.expected_limits(
            theta_true=theta_true,
            theta_ranges=theta_ranges,
            mode="histo",
            hist_vars=[ unicode(str(asymptotic['hist_vars'])) ],
            include_xsec=bool_xsec,
            resolutions=resolutions,
            luminosity=uselumi)

        if (bool_xsec==True):
            np.save('/madminer/rates/histo.npy',[p_values_expected_histo, best_fit_expected_histo])
        else:
            np.save('/madminer/rates/histo_kin.npy',[p_values_expected_histo, best_fit_expected_histo])




    for bool_xsec in xsec:

        # method ML +save
        if( method in ['alice','alices','cascal','carl','rolr', 'rascal'] ):
            theta_grid, p_values_expected_method, best_fit_expected_method = limits.expected_limits(
            theta_true=theta_true,
            theta_ranges=theta_ranges,
            mode="ml",
            model_file=eval_folder_path+'/'+method,
            include_xsec=bool_xsec,
            resolutions=resolutions,
            luminosity=uselumi)
            resultsdir = '/madminer/results/'+method+'/ml'
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
            theta_ranges=theta_ranges,
            mode="histo",
            model_file= eval_folder_path+'/'+method, 
            include_xsec=bool_xsec,
            resolutions=resolutions,
            luminosity=uselumi)

            resultsdir = '/madminer/results/'+method+'/histo'
            if not os.path.isdir(resultsdir):
                os.makedirs(resultsdir)
            if (bool_xsec==True):
                np.save(resultsdir+'/'+method+'.npy',[p_values_expected_method, best_fit_expected_method])
            else:
                np.save(resultsdir+'/'+method+'_kin.npy',[p_values_expected_method, best_fit_expected_method])
            


# FISHER INFO
if( bool(inputs['fisher_information']['bool']) and (method in ['sally']) ):
  
  uselumi = float(inputs['uselumi'])
  fisher_input = inputs['fisher_information']
  fisher = FisherInformation(h5_file, include_nuisance_parameters=False)
  theta_true = [ float(fisher_input['theta_true'][0]),\
                   float(fisher_input['theta_true'][1]), \
                   float(fisher_input['theta_true'][2]) ]

  fisher_information, _ = fisher.calculate_fisher_information_full_detector(
    theta=[0.,0.,0.],
    model_file='/madminer/models/sally/sally',
    luminosity=30000.)
        
  # contourplot.savefig('/madminer/plots/plot_fisher.png')




# EVALUATE: TRAIN VS TEST

#perform the test + evaluation score acconding to method
if(method in ['alice','alices','cascal','carl','rolr', 'rascal']):

  #generate test data
  generate_test_data_ratio(method)

  forge = ParameterizedRatioEstimator() 
  forge.load(eval_folder_path+'/'+method)  #'methods/alices'

  theta_grid=np.load('/madminer/rates/grid.npy')
  xs_grid=np.load('/madminer/rates/xs_grid.npy')
  redo_limits=False

 
  # From Asymptotic Limits: _calculate_xsecs
  limits = AsymptoticLimits(h5_file)
  xs_limits = limits._calculate_xsecs([theta_true],test_split=float(inputs['test_split']))[0]
  print ("AsymptoticLimits (_calculate_xsecs): ", xs_limits)

  # From Sample Augmenter cross_sections
  sa = SampleAugmenter(h5_file, include_nuisance_parameters=False)
  _,xs_sa,_=sa.cross_sections(theta=sampling.morphing_point(theta_true))
  print ("SampleAugmenter (cross_sections) : ", xs_sa[0])

  # From Sample Augmenter: weighted_events
  _,w=sa.weighted_events(theta='sm')
  xs_we=sum(w)
  print ("SampleAugmenter (weighted_events): ", xs_we)

  # From Sample Augmenter: xsecs
  xs_xsecs,_=sa.xsecs(thetas=[theta_true], events='train',  test_split=float(inputs['test_split']))
  print ("SampleAugmenter (xsecs) : ", xs_xsecs[0])

  
  #n_test = int(inputs['n_samples']['test'])
  #data,_,_=sa.sample_train_plain(theta=sampling.morphing_point(theta_true),n_samples=n_test) 
  
  #Calc expected number of events 
  _,xs,_=sa.cross_sections(theta=sampling.morphing_point(theta_true))
  nevents = uselumi*xs[0]


  #Get LLR
  out_llr=[]
  out_llr_raw=[]
  out_llr_rescaled=[]
  out_llr_substracted=[]
  out_pval=[]
  out_theta=[]

  for i,theta_element in enumerate(theta_grid):
    
    llr,score=forge.evaluate_log_likelihood_ratio(
        theta=np.array([theta_element]),
        x='/madminer/test/'+method+'/x_test.npy',
        evaluate_score=True,
        test_all_combinations=True,
    )
    
    llr_raw= sum(llr[0])/n_test
    llr_rescaled= nevents*llr_raw 
    
    out_llr.append(llr)
    out_llr_raw.append(llr_raw)
    out_llr_rescaled.append(llr_rescaled)
    out_theta.append(theta_element)

  llrmin = np.argmin(out_llr_rescaled)
  out_llr_substracted,_=limits._subtract_ml(out_llr_rescaled) 
  out_pval=limits.asymptotic_p_value(out_llr_substracted) 
    
  #save to files
  print('Saving Raw mean -2 log r  to file: ', '/madminer/results/'+method+'/llr_raw.npy')
  np.save('/madminer/results/'+method+'/llr_raw.npy', out_llr_raw)

  print('Saving Rescaled -2 log r  to file: /madminer/results/'+method+'/llr_rescaled.npy')
  np.save('/madminer/results/'+method+'/llr_rescaled.npy', out_llr_rescaled)

  print('Saving Raw mean Min-subtracted -2 log r to file: ', '/madminer/results/'+method+'/llr_substracted.npy')
  np.save('/madminer/results/'+method+'/llr_substracted.npy', out_llr_substracted)

  print('Saving p-values  to file: ', '/madminer/results/'+method+'/p_values.npy')
  np.save('/madminer/results/'+method+'/p_values.npy', out_pval)

  print('Saving score  to file: ', '/madminer/results/'+method+'/score.npy')
  np.save('/madminer/results/'+method+'/score.npy', score)


if(method in ['sally','sallino']):

  #generate test data
  generate_test_data_score(method)

  forge = LikelihoodEstimator()
  forge.load(eval_folder_path+'/'+method)  #'methods/alices'

  theta_grid=np.load('/madminer/rates/grid.npy')
  xs_grid=np.load('/madminer/rates/xs_grid.npy')
  redo_limits=False

 
  # From Asymptotic Limits: _calculate_xsecs
  limits = AsymptoticLimits(h5_file)
  xs_limits = limits._calculate_xsecs([theta_true],test_split=float(inputs['test_split']))[0]
  print ("AsymptoticLimits (_calculate_xsecs): ", xs_limits)

  # From Sample Augmenter cross_sections
  sa = SampleAugmenter(h5_file, include_nuisance_parameters=False)
  _,xs_sa,_=sa.cross_sections(theta=sampling.morphing_point(theta_true))
  print ("SampleAugmenter (cross_sections) : ", xs_sa[0])

  # From Sample Augmenter: weighted_events
  _,w=sa.weighted_events(theta='sm')
  xs_we=sum(w)
  print ("SampleAugmenter (weighted_events): ", xs_we)

  # From Sample Augmenter: xsecs
  xs_xsecs,_=sa.xsecs(thetas=[theta_true], events='train',  test_split=float(inputs['test_split']))
  print ("SampleAugmenter (xsecs) : ", xs_xsecs[0])

  
  #n_test = int(inputs['n_samples']['test'])
  #data,_,_=sa.sample_train_plain(theta=sampling.morphing_point(theta_true),n_samples=n_test) 
  
  #Calc expected number of events 
  _,xs,_=sa.cross_sections(theta=sampling.morphing_point(theta_true))
  nevents = uselumi*xs[0]


  #Get LLR
  out_llr=[]
  out_llr_raw=[]
  out_llr_rescaled=[]
  out_llr_substracted=[]
  out_pval=[]
  out_theta=[]

  for i,theta_element in enumerate(theta_grid):
    
    llr,_=forge.evaluate_score(
        theta=np.array([theta_element]),
        x='/madminer/test/'+method+'/x_test.npy'
    )
    
    llr_raw= sum(llr[0])/n_test
    llr_rescaled= nevents*llr_raw 
    
    out_llr.append(llr)
    out_llr_raw.append(llr_raw)
    out_llr_rescaled.append(llr_rescaled)
    out_theta.append(theta_element)

  llrmin = np.argmin(out_llr_rescaled)
  out_llr_substracted,_=limits._subtract_ml(out_llr_rescaled) 
  out_pval=limits.asymptotic_p_value(out_llr_substracted) 
    
  #save to files
  print('Saving Raw mean -2 log r  to file: ', '/madminer/results/'+method+'/llr_raw.npy')
  np.save('/madminer/results/'+method+'/llr_raw.npy', out_llr_raw)

  print('Saving Rescaled -2 log r  to file: /madminer/results/'+method+'/llr_rescaled.npy')
  np.save('/madminer/results/'+method+'/llr_rescaled.npy', out_llr_rescaled)

  print('Saving Raw mean Min-subtracted -2 log r to file: ', '/madminer/results/'+method+'/llr_substracted.npy')
  np.save('/madminer/results/'+method+'/llr_substracted.npy', out_llr_substracted)

  print('Saving p-values  to file: ', '/madminer/results/'+method+'/p_values.npy')
  np.save('/madminer/results/'+method+'/p_values.npy', out_pval)

  print('Saving score  to file: ', '/madminer/results/'+method+'/score.npy')
  np.save('/madminer/results/'+method+'/score.npy', score)









  # evaluation = inputs['evaluation']['theta_each']
  # theta_each = np.linspace( float(evaluation['start']), float(evaluation['stop']), int(evaluation['num']) ) 
  # theta0, theta1 = np.meshgrid(theta_each, theta_each)
  # theta_grid = np.vstack((theta0.flatten(), theta1.flatten())).T #numtidim
  # np.save('/madminer/data/test/theta_grid.npy', theta_grid)

  # log_r_hat, score_theta0, _ = forge.evaluate(
    #     theta0_filename='/madminer/data/test/theta_grid.npy',
    #     x='/madminer/data/test/x_test.npy',
    #     evaluate_score=inputs['evaluation']['evaluate_score']


if(method in ['alice2','alices2','cascal2','carl2','rolr2', 'rascal2']):
  forge = DoubleParameterizedRatioEstimator() 
  forge.load(eval_folder_path+'/'+method)  #'methods/alices'


# #?
# theta_denom = np.array([[0.,0.]])
# np.save('/madminer/data/test/theta_ref.npy', theta_denom)

# print('you need to update v0.3.0 to forge.evaluate_likelihood() or similar ')

# if( method  in ['alice',  'alices',  'carl',  'nde', 'rascal',  'rolr',  'scandal'] ):
    
#   #make and save grid
#   
#   )
#   with open('/madminer/data/test/log_r_hat_'+method+'.npy', "w+") as f: #create file
#       np.save(file='/madminer/data/test/log_r_hat_'+method, arr=log_r_hat)
    
#   with open('/madminer/data/test/score_theta0_'+method+'.npy', "w+") as g: #create file
#       np.save(file='/madminer/data/test/score_theta0_'+method, arr=score_theta0)
    

#   #plots
#   if( bool(inputs['plots']['activate']) ):
        
#       bin_size = theta_each[1] - theta_each[0]
#       edges = np.linspace(theta_each[0] - bin_size/2, theta_each[-1] + bin_size/2, len(theta_each)+1)

#       fig = plt.figure(figsize=(6,5))
#       ax = plt.gca()

#       expected_llr = np.mean(log_r_hat,axis=1)
#       best_fit = theta_grid[np.argmin(-2.*expected_llr)]

#       cmin, cmax = np.min(-2*expected_llr), np.max(-2*expected_llr)
            
#       pcm = ax.pcolormesh(edges, edges, -2. * expected_llr.reshape((21,21)),
#                           norm=matplotlib.colors.Normalize(vmin=cmin, vmax=cmax),
#                           cmap='viridis_r')
#       cbar = fig.colorbar(pcm, ax=ax, extend='both')

#       plt.scatter(best_fit[0], best_fit[1], s=80., color='black', marker='*')

#       plt.xlabel(r'$\theta_0$')
#       plt.ylabel(r'$\theta_1$')
#       cbar.set_label(r'$\mathbb{E}_x [ -2\, \log \,\hat{r}(x | \theta, \theta_{SM}) ]$')

#       plt.tight_layout()
#       plt.savefig('/madminer/plots/expected_llr_'+method+'.png')  


# if( method  in ['alice2', 'alices2', 'carl2', 'rascal2', 'rolr2' ] ):
#   print('evaluation for this method is not yet implemented')
#   pass 

#   #make and save grid
#   evaluation = inputs['evaluation']['theta_each']
#   theta_each = np.linspace( float(evaluation['start']), float(evaluation['stop']), int(evaluation['num']) ) 
#   theta0, theta1 = np.meshgrid(theta_each, theta_each)
#   theta_grid = np.vstack((theta0.flatten(), theta1.flatten())).T #numtidim
#   np.save('/madminer/data/test/theta_grid.npy', theta_grid)

#   log_r_hat, score_theta0, score_theta1 = forge.evaluate(
#       theta0_filename='/madminer/data/test/theta_grid.npy',
#       theta1_filename='/madminer/data/test/theta_grid.npy', #TO DO !
#       x='/madminer/data/test/x_test.npy',
#       evaluate_score=False
#   )
#   with open('/madminer/data/test/log_r_hat_'+method+'.npy', "w+") as f: #create file
#       np.save(file='/madminer/data/test/log_r_hat_'+method, arr=log_r_hat)        


# if( method  in ['sally', 'sallino'] ):
#   t_hat = forge.evaluate(
#       x='/madminer/data/samples/x_test.npy'
#   )
#   with open('/madminer/data/test/t_hat_'+method+'.npy', "w+") as f: #create file
#       np.save(file='/madminer/data/test/t_hat_'+method, arr=t_hat)

#   #plots
#   if( bool(inputs['plots']['activate']) ):
#       x = np.load('data/samples/x_test.npy')
#       fig = plt.figure(figsize=(10,4))

#       for i in range(2):
#           ax = plt.subplot(1,2,i+1)
#           sc = plt.scatter(x[::10,0], x[::10,1], c=t_hat[::10,i], s=10., cmap='viridis', vmin=-0.8, vmax=0.4)
#           cbar = plt.colorbar(sc)
#           cbar.set_label(r'$\hat{t}_' + str(i) + r'(x | \theta_{ref})$')
#           plt.xlabel(r'$p_{T,j1}$ [GeV]')
#           plt.ylabel(r'$\Delta \phi_{jj}$')
#           plt.xlim(10.,400.)
#           plt.ylim(-3.15,3.15)
#           plt.tight_layout()
#           plt.savefig('/madminer/plots/t_hat_'+method+'.png') 


