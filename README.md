![Build status][travis-build-status]

# Madminer Deployment using MLFlow
This repository intends to create a full deployment of 
[MadMiner](https://github.com/johannbrehmer/madminer) (by Johann Brehmer, Felix 
Kling, and Kyle Cranmer) that runs on the OpenSource ML platform, 
[MLFlow](https://www.mlflow.org)

## To deploy in Kubernetes
The inputs for `configurate` are mounted as a ConfigMap. This is deployed
into the cluster with 
```bash
kubectl create -f kube/input_configmap.yml
```

It writes intermediate data to a persistent volume. For testing on a single
node cluster, you can create the persistent volume claim with
```bash
kubectl create -f kube/pv.yml
kubectl create -f kube/pvc.yml
```

## Build the Physics Docker Image
Build the docker image for the MadGraph physics container with
```bash
docker login
cd docker-images/docker-madminer-physics
docker build -t <username>/docker-madminer-physics:mlflow .
```
Where <username> is your dockerhub username. MLFlow is going to create a docker
image derived off this image and push it to dockerhub so the image needs to be 
tagged with your username.

## Run the physics package in MLFlow
MLFlow's workflow is controlled by the file `mlflow/MLProject`. It defines the
image to be used along with a fake parameter that represents the 
hyperparameters that could be set to adjust the training.

The file `mlflow/kubernetes_job_template.yaml` is a template for the job
that will be launched into the cluster. It shows how the configmap and 
persistent volume will be mounted into each pod.

```bash
pip install mlflow mlflow run mlflow -P alpha=0.5 --backend kubernetes --backend-config mlflow/kubernetes_config.json

```

 
# Madminer deployment using REANA, yadage and docker containerization

## About
This repository intends to create a full deployment of [MadMiner][madminer-repo]
(by Johann Brehmer, Felix Kling, and Kyle Cranmer) that parallelizes its bottlenecks, is reproducible 
and facilitates the use of the tool in the community.

To achieve this we have generated a workflow using [yadage][yadage-repo] (by Lukas Heinrich)
and a containerization of the software dependencies in several docker images.
The pipeline can be run with [REANA][reana-webpage], a data analysis platform.

This repo includes the workflow for the physics processing (config, generation of events with MadGraph, Delphes)
and the machine learning processing (configuration, sampling, training) in a modular way.
This means that each of this parts has its own workflow setup so that the user can mix-match.
For instance, once the physics processes are run, one can play with different hyperparameters
or samplings in the machine learning part, without having to re-run MadGraph again.

Please, refer to the links for more information and tutorials about
[MadMiner][madminer-docs] ([tutorial][madminer-tutorial]) and
[yadage][yadage-docs] ([tutorial][yadage-tutorial])


## Docker images
MadMiner is a set of complex tools with many steps. For that reason, we considered better to split
the software dependencies and workflow code in two Docker images,
plus another one containing the latest MadMiner library version. 
All of the official images are hosted in the [madminertool][madminer-dockerhub] DockerHub.

- [madminertool/docker-madminer][madminer-raw-docker-image]:
contains only latest version of MadMiner.
- [madminertool/docker-madminer-physics][madminer-ph-docker-image]:
contains the code necessary to configure, generate and process events according to MadMiner.
You will also find the software dependencies in the directory `/home/software`. [Dockerfile][madminer-ph-docker-file].
- [madminertool/docker-madminer-ml][madminer-ml-docker-image]:
contains the code necessary to configure, train and evaluate in the MadMiner framework. [Dockerfile][madminer-ml-docker-file].

To pull any of the images and see its content:
```bash
docker pull madminertool/<image-name>
docker run -it madminertool/<image-name> bash
<container-id>#/home $ ls
```

_The point of this repository is to make the life easier for the users so they won't need
to figure out themselves the arguments of the scripts on `/home/code/` nor how to input new observables.
The whole pipeline will be automatically generated when they follow the steps in the sections below.
Also they will have the chance to input their own parameters, observables, cuts etc.
without even needing to pull the Docker images._


## Dependencies and workflows
Dependency installation depends on how you want to run the workflow: using a REANA cluster, or locally using yadage.


### Deploy with REANA
To deploy Madminer locally using [REANA][reana-webpage], use _VirtualBox_ as emulator and _Minikube_ 
as container orchestrator in order to simulate a local cluster. Please refer to the **version 0.7.0**
[new REANA deployment documentation][reana-deploy-docs] for more details.

If you have access to a remote REANA cluster, introduce only the credentials as shown below, to generate 
the following combined-workflow:

![image of the workflow](docs/images/workflow-all.png)

To introduce the credentials, and start the workflow:
```bash
$ source ~/.virtualenvs/reana/bin/activate
# A) Enter credentials for a remote-cluster
(reana) $ export REANA_ACCESS_TOKEN = [..]
(reana) $ export REANA_SERVER_URL = [..]
# B) Enter credentials for a minikube local-cluster
(reana) $ eval $(reana-dev setup-environment)
# Check connectivity to the cluster
(reana) $ reana-client ping
# Create the analysis from within the 'reana' folder
(reana) $ cd reana
(reana) $ reana-client create -n madminer-workflow
(reana) $ export REANA_WORKON=madminer-workflow
(reana) $ reana-client upload ./inputs/input.yml
(reana) $ reana-client upload ./workflows/yadage/workflow.yml
(reana) $ reana-client upload ./workflows/yadage/steps.yml
(reana) $ reana-client start
(reana) $ reana-client status
```

It might take some time to finish depending on the job and the cluster. Once it does, list and download the files:
```bash
(reana) $ reana-client ls
(reana) $ reana-client download <path/to/file/on/reana/workon>
```

The command `reana-client ls` will display the folders containing the results from each step.
You can download any intermediate result you are interested in (for example `combine/combined_delphes.h5`,
`evaluating_0/Results.tar.gz` or all the plots available in `plotting/`).


### Deploy with Yadage (locally)
Install Yadage and its dependencies to visualize workflows:
```bash
pip install yadage[viz]
```

Also, you would need the `graphviz` package. Check it was successfully installed by running:
```bash
yadage-run -t from-github/testing/local-helloworld workdir workflow.yml -p par=World
```

It should see a log entry like the following. Refer to the Yadage links above for more details:
```
2019-01-11 09:51:51,601 | yadage.utils | INFO | setting up backend multiproc:auto with opts {}
```

For the first run, we recommend using our default `input.yml` files, decreasing both `njobs` and `ntrainsamples` for speed.

Finally, move to the directory of the example and run:
```bash
sudo rm -rf workdir && yadage-run workdir workflow.yml \
    -p inputfile='"inputs/input.yml"' \
    -p njobs="6" \
    -p ntrainsamples="2" \
    -d initdir=$PWD \
    --visualize
```

To run again the command you must first remove the `workdir` folder:
```bash
rm -rf workdir/
```

**What is every element in the command?**
- `workdir`: specifies the new directory where all the intermediate and output files will be saved.
- `workflow.yml`: file connecting the different stages of the workflow. It must be placed in the working directory.
- `-p <param>`: all the parameters. In this case:
    - `inputfile` has a configuration of all the parameters and options.
    - `njobs`: number of maps in the Physics workflow.
    - `ntrainsamples` number to scale the training sample process.
- `-d initdir`: specifies the directory where the the workflow is initiated.
- `--visualize`: specifies the creation of a workflow image.


## Analysis structure

# Running MadMiner With MLFlow
As an experiment, we are testing MadMiner using the open source Maching Learning
platform, [MLFlow](https://mlflow.org/)

## Configure MLFlow
We are using the new kubernetes support for MLFlow. We found a little 
deficiency in the functionality that makes testing harder than it should be
and have proposed a pull request to fix it. In the mean time we will need to 
use our fork of MLFlow.

Building the MLFlow command line is easy. There are a few extra steps to build 
the UI, so we will jump through some hoops to skip that step


*First create two virtualenvs*
Make sure you have python 3 installed.

```bash
virtualenv ~/.virtualenvs/madminer
source ~/.virtualenvs/mlflow/bin/activate
```

*Launch the MLFlow Tracking UI*
In a terminal window activate the ui's virtual environment, install the 
published mlflow and launch

```bash
source ~/.virtualenvs/mlflow/bin/activate
pip install mlflow
mlflow ui
```

*Checkout The Fork and Install*
```bash
source ~/.virtualenvs/madminer/bin/activategit clone https://github.com/scailfin/mlflow.git
cd mlflow
git checkout  k8s_mflow_tracking_url
```


Now install into your virtual environment with 
```bash
python setup.py install
```

*Running the Configurate Job*
In this repo you will build the madminer physics docker image

```bash
cd docker-images/docker-madminer-physics
```

You need to make sure there is a copy of `MG5_aMC_v2_6_2` in this directory

```bash
docker build -t <your dockerhub user>/docker-madminer-physics:mlflow .       
```

*Create Kubernetes Objects*
We have three kubernetes objects that the job depends on 
* input_configmap.yml - This contains a yaml file that controls the configurate
job.
* pv - Creates a persistent volume that the different jobs share and allow you 
to access the outputs
* pvc - Creates persistent volume claim that will bind each job to the volume

Edit `kube/pv` so the hostpath points to an absolute directory on your desktop

Then deploy the objects to your cluster with 
```bash
kubectl create -f kube/pv.yml
kubectl create -f kube/pvc.yml
kubectl create -f kube/input_configmap.yml
``` 

*Update the MLProject*
The folder `mlflow` contains the actual project we will run. Edit 
`mlflow/kubernetes_config.json` so the `repository-uri` has your docker hub 
username.

*Ready to Run your First Job!*
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000  
export K8S_MLFLOW_TRACKING_URI=http://host.docker.internal:5000   
mlflow run mlflow -P alpha=0.5 --backend kubernetes --backend-config mlflow/kubernetes_config.json
```

You should see the job build a derived docker image, push to your dockerhub and 
then run. The steps will show up in the ui.

# Running MadMiner With MLFlow
As an experiment, we are testing MadMiner using the open source Maching Learning
platform, [MLFlow](https://mlflow.org/)

## Configure MLFlow
We are using the new kubernetes support for MLFlow. We found a little 
deficiency in the functionality that makes testing harder than it should be
and have proposed a pull request to fix it. In the mean time we will need to 
use our fork of MLFlow.

Building the MLFlow command line is easy. There are a few extra steps to build 
the UI, so we will jump through some hoops to skip that step


*First create two virtualenvs*
Make sure you have python 3 installed.

```bash
virtualenv ~/.virtualenvs/madminer
source ~/.virtualenvs/mlflow/bin/activate
```

*Launch the MLFlow Tracking UI*
In a terminal window activate the ui's virtual environment, install the 
published mlflow and launch

```bash
source ~/.virtualenvs/mlflow/bin/activate
pip install mlflow
mlflow ui
```

*Checkout The Fork and Install*
```bash
source ~/.virtualenvs/madminer/bin/activategit clone https://github.com/scailfin/mlflow.git
cd mlflow
git checkout  k8s_mflow_tracking_url
```


Now install into your virtual environment with 
```bash
python setup.py install
```

*Running the Configurate Job*
In this repo you will build the madminer physics docker image

```bash
cd docker-images/docker-madminer-physics
```

You need to make sure there is a copy of `MG5_aMC_v2_6_2` in this directory

```bash
docker build -t <your dockerhub user>/docker-madminer-physics:mlflow .       
```

*Create Kubernetes Objects*
We have three kubernetes objects that the job depends on 
* input_configmap.yml - This contains a yaml file that controls the configurate
job.
* pv - Creates a persistent volume that the different jobs share and allow you 
to access the outputs
* pvc - Creates persistent volume claim that will bind each job to the volume

Edit `kube/pv` so the hostpath points to an absolute directory on your desktop

Then deploy the objects to your cluster with 
```bash
kubectl create -f kube/pv.yml
kubectl create -f kube/pvc.yml
kubectl create -f kube/input_configmap.yml
``` 

*Update the MLProject*
The folder `mlflow` contains the actual project we will run. Edit 
`mlflow/kubernetes_config.json` so the `repository-uri` has your docker hub 
username.

*Ready to Run your First Job!*
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000  
export K8S_MLFLOW_TRACKING_URI=http://host.docker.internal:5000   
mlflow run mlflow -P alpha=0.5 --backend kubernetes --backend-config mlflow/kubernetes_config.json
```

You should see the job build a derived docker image, push to your dockerhub and 
then run. The steps will show up in the ui.

The `-P alpha=0.5` is just a demonstration of passing parameters into the run
that get tracked by the ui.
