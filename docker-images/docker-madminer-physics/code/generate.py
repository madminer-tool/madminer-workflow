#!/usr/bin/python

import sys
from madminer import MadMiner
from pathlib import Path


##########################
#### Argument parsing ####
##########################

num_jobs = int(sys.argv[1])
config_file = str(sys.argv[2])
output_dir = Path(sys.argv[3])

project_dir = Path(__file__).parent.parent

card_dir = str(project_dir.joinpath('code', 'cards'))
madg_dir = str(project_dir.joinpath('software', 'MG5_aMC_v2_6_7'))

logs_dir = str(output_dir.joinpath('logs'))
proc_dir = str(output_dir.joinpath('mg_processes'))


##########################
### Load configuration ###
##########################

miner = MadMiner()
miner.load(config_file)

benchmarks = [str(i) for i in miner.benchmarks]
num_benchmarks = len(benchmarks)
print(f'Benchmarks {benchmarks}')


##########################
### Define run wrapper ###
##########################

def madminer_run_wrapper(sample_benchmarks, run_type):
    """
    Wraps the MadMiner run_multiple function

    :param sample_benchmarks: list of benchmarks
    :param run_type: either 'background' or 'signal'
    """

    if run_type == 'background':
        is_background = True
    elif run_type == 'signal':
        is_background = False
    else:
        raise ValueError('Invalid run type')

    miner.run_multiple(
        is_background=is_background,
        only_prepare_script=True,
        sample_benchmarks=sample_benchmarks,
        mg_directory=madg_dir,
        mg_process_directory=proc_dir + '/' + run_type,
        proc_card_file=card_dir + f'/proc_card_{run_type}.dat',
        param_card_template_file=card_dir + '/param_card_template.dat',
        run_card_files=[card_dir + f'/run_card_{run_type}.dat'],
        pythia8_card_file=card_dir + '/pythia8_card.dat',
        log_directory=logs_dir + '/' + run_type,
        python2_override=True,
    )

    # Create files to link benchmark_i to run_i.sh
    for i in range(num_jobs):
        index = i % num_benchmarks
        file_path = proc_dir + f'/{run_type}/madminer/cards/benchmark_{i}.dat'

        with open(file_path, "w+") as f:
            f.write("{}".format(benchmarks[index]))

        print('generate.py', i, benchmarks[index])


###########################
##### Run with signal #####
###########################

# Sample benchmarks from already stablished benchmarks in a democratic way
initial_list = benchmarks[0 : (num_jobs % num_benchmarks)]
others_list = benchmarks * (num_jobs // num_benchmarks)
sample_list = initial_list + others_list

madminer_run_wrapper(sample_benchmarks=sample_list, run_type='signal')


###########################
### Run with background ###
###########################

# Currently not used
# sample_list = ['sm' for i in range(num_jobs)]
# madminer_run_wrapper(sample_benchmarks=sample_list, run_type='background')
