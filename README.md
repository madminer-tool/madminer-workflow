[![Build status](https://travis-ci.com/irinaespejo/workflow-madminer.svg?branch=master)](https://travis-ci.org/irinaespejo)
[![MIT Licence](https://badges.frapsoft.com/os/mit/mit.png?v=103)](https://opensource.org/licenses/mit-license.php)

# Madminer deployment using yadage and docker containerization

## About
This repository is a  is to create a full deployment of [MadMiner](https://github.com/johannbrehmer/madminer) (by Johann Brehmer, Felix Kling, and Kyle Cranmer) that parallelizes its bottlenecks, is reproducible and it facilitates the use of the tool in the community. 
To achieve this we have generated a workflow using [yadage](https://github.com/yadage/yadage)
(by Lukas Heinrich) and a containerization of the software dependencies in a *docker image*.

At the moment only the initial steps in the Madminer workflow are implemented, that is those regarding the generation of events and  the simulators MadGraph+Pythia and Delphes. The implementation of the Machine Learning part is work in progress.

Please, refer to the links for more information and tutorials about MadMiner 
https://madminer.readthedocs.io/en/latest/index.html
https://github.com/johannbrehmer/madminer/blob/master/examples/tutorial/tutorial_1.ipynb
and yadage
https://yadage.readthedocs.io/en/latest/
https://yadage.github.io/tutorial/

## How to install the dependencies

###Local yadage-pure deployment

1) Install yadage and the dependencies to visualize workflows [viz]
```bash
  pip install yadage[viz]
```
Also, you need the graphviz package
Check it was succesful by running:
```bash
  yadage-run -t from-github/testing/local-helloworld workdir workflow.yml -p par=World
```
It should output lines similar to this one `2019-01-11 09:51:51,601 |         yadage.utils |   INFO | setting up backend multiproc:auto with opts {}` and no errors. Refer to the yadage links above for more details.

2) Pull the docker image irinahub/docker-madminer
This step is optional because the image will be pulled automatically in the Usage steps. However you may want to familirize yourself with the image and its contents. If you don't have docker installed follow https://docs.docker.com/install/
To pull the image run
```bash
  docker pull irinahub/docker-madminer
```
To run a container and interact with the bash run
```bash
  docker run -it irinahub/docker-madminer bash
```
For more details about the image visit https://github.com/irinaespejo/docker-madminer
*The point of this repository is to make the life easy for the user so you won't need to figure out yourself the arguments of the scripts on /home/code/ nor how to input new observables. The whole pipeline will be automatically generated when you follow the steps in the section Usage and you will have the chance to input your own parameters, observables, cuts etc. without messing with the docker image.*

### Local REANA deployment
To deploy Madminer locally using [REANA](http://www.reana.io/) we will use Minikube as emulator for a cluster. Here we will give basic steps to set up Minikube as a `reana-cluster`. Please refer to https://reana-cluster.readthedocs.io/en/latest/gettingstarted.html  for more details. You will need to have [Minikube](https://kubernetes.io/docs/tasks/tools/install-minikube/) v0.28.2 and [Kubernettes](https://kubernetes.io/docs/tasks/tools/install-kubectl/) v1.11.2  installed.

Run

```bash
  minikube start --kubernetes-version="v1.11.2" --vm-driver kvm2
```
(`-vm--driver`  may depend on the OS)

```bash
virtualenv ~/.virtualenvs/myreana
```

```bash
source ~/.virtualenvs/myreana/bin/activate
```

```bash
(myreana) pip install reana-cluster
```

```bash
(myreana) reana-cluster init
```

```bash
(myreana) reana-cluster status
```

```bash
(myreana) eval $(reana-cluster env --include-admin-token)
```
with this you have set up a local `reana-cluster`. Now let's set up the` reana-client` where you will run the analysis. Please refer to https://reana-client.readthedocs.io/en/latest/gettingstarted.html  for more details.

```bash
(myreana) pip install reana-client
```
check connectivity to `reana-cluster`
```bash
(myreana) reana-client ping
```
Create the analysis
```bash
(myreana) reana-client create -n my-analysis
```

```bash
(myreana) export REANA_WORKON=my-analysis
```

```bash
(myreana) reana-client upload ./data
```

```bash
(myreana) reana-client start
```

```bash
(myreana) reana-client status
```
it might take a long time to finish, once it does download files
```bash
(myreana) reana-client download <path/to/file/on/reana/workon>
```


## Usage
For the first run we recommend using our default files `input.yml` and `input_delphes.yml`.
To generate the following workflow 

![image of the workflow](images/yadage_workflow_instance.png)

run 
```bash
  cd workflow/
  yadage-run workdir workflow.yml -p inputfile='"input.yml"' -p njobs="10" -p inputdelphes='"input_delphes.yml"' -d initdir=$PWD --visualize
```
to run again the command you must first remove workdir `rm -rf workdir/`
>what is every element in the command?
	- `workdir` is creating a new dir where all the intermediate and output files will be saved.
	- `workflow.yml` is the file that connects the different stages of the workflow, it must be placed in the working directory
	- all the parameters are preceed by `-p`: `njobs` is the number of maps in the workflow, `inputfile` has the parameters and `input_delphes.yml` for observables and cuts.
	- `-d initdir=$PWD` initializes the workflow in the present directory
	- `--visualize` generates an image of the workflow

## Analysis structure
### 1. Analysis code
configurate.py - Put together inputs and initialize
	- Initializes MadMiner, add parameters, add benchmarks, set benchmarks from morphing and save.
	`python code/configurate.py` 
	
generate.py - Generate events scripts
	- Prepare scripts for MadGraph for background and signal events based on previous optimization.
	`python code/generate.py {njobs} {h5_file}` where `{njobs}` is the initial parameter  `njobs` and `{h5_file}` is a file generated in configurate.py with the MadMiner configuration.

delphes.py - Run Delphes
	- Pass the events through Delphes, add observables and cuts, save.
	 `python code/delphes.py {h5_file} {event_file} {input_file}` where  `{h5_file}` is the same file as above  `{event_file}` is the file  `tag_1_pythia8_events.hepmc.gz` and `{input_file}{input_file}` is the initial `input_delphes.yml`

### 2. Anaylsis workflow
Without taking into account the inputs ans the map-reduce the general strcture of the workflow is the following

				+--------------+
				|  Configurate |
				+--------------+
					   |
					   |
					   v
				+--------------+
				|   Generate   |
				+--------------+
	   				   |
					   |
					   v
				+--------------+
				|   MG+Pythia  |
				+--------------+
					   |
					   |
					   v
				+--------------+
				|   Delphes    |
				+--------------+
					   |
					   |
					   v
				+-----------------+
				| To be continued |
				+-----------------+

