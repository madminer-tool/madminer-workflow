from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
#import matplotlib
#from matplotlib import pyplot as plt
#%matplotlib inline
import sys 
import yaml
import inspect
from madminer.core import MadMiner
from madminer.plotting import plot_2d_morphing_basis
from madminer.sampling import combine_and_shuffle
from madminer.sampling import SampleAugmenter
from madminer.sampling import benchmark, benchmarks
from madminer.sampling import morphing_point, morphing_points, random_morphing_points


mg_dir = '/home/software/MG5_aMC_v2_6_2'

miner = MadMiner()#(debug=False)

input_file = str(sys.argv[1])
print('inputfile:  ',input_file)

########### ADD  parameters and benchmarks from input file

with open(input_file) as f:
    # use safe_load instead load
    dict_all = yaml.safe_load(f)

#get default values of miner.add_parameters()
default_arr = inspect.getargspec(miner.add_parameter)
default = dict(zip(reversed(default_arr.args), reversed(default_arr.defaults)))


#ADD PARAMETERS
for parameter in dict_all['parameters']:
    #format range_input to tuple
    range_input = parameter['parameter_range']
    range_tuple = map(float, range_input.replace('(','').replace(')','').split(','))
   
    miner.add_parameter(
    lha_block=parameter['lha_block'], #required
    lha_id=parameter['lha_id'], #required
    parameter_name=parameter.get('parameter_name', default['parameter_name']), #optional
    morphing_max_power=int( parameter.get('morphing_max_power', default['morphing_max_power']) ), #optional
    param_card_transform=parameter.get('param_card_transform',default['param_card_transform']),  #optional
    parameter_range=range_tuple #optional
    )

n_parameters = len(dict_all['parameters'])


#ADD BENCHMARKS
for benchmark in dict_all['benchmarks']:
    
    dict_of_parameters_this_benchmark = dict()
    
    for i in range(1, n_parameters+1):

        try:
            #add to the dictionary: key is parameter name, value is value
            dict_of_parameters_this_benchmark[ benchmark['parameter_name_'+str(i)] ] = float(benchmark['value_'+str(i)])
        
        except KeyError as e:
            print('Number of benchmark parameters does not match number of global parameters in input file')
            raise e
    
    #add       
    miner.add_benchmark(
    dict_of_parameters_this_benchmark,
    benchmark['name']
    )

###########

#SET morphing
settings = dict_all['set_morphing']
miner.set_morphing(
    include_existing_benchmarks=True,
    max_overall_power=int(settings['max_overall_power'])
)


#fig = plot_2d_morphing_basis(
#    miner.morpher,
#    xlabel=r'$c_{W} v^2 / \Lambda^2$',
#    ylabel=r'$c_{\tilde{W}} v^2 / \Lambda^2$',
#    xrange=(-10.,10.),
#    yrange=(-10.,10.)
#)

miner.save('/home/data/madminer_example.h5')

