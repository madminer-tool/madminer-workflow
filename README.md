# Madminer workflow


## About
This repository serves as wrapper around [Madminer physics workflow][madminer-workflow-ph]
and [Madminer ML workflow][madminer-workflow-ml] to construct a single, linked workflow
to be executed in [REANA][reana-website].

Both workflows are defined as GIT submodules in this repository. Submodules allow us
to combine contents from different repositories when contents of both are necessary
to perform some complex operation, but they are, indeed, different projects.


## Workflow definition
The workflow specification is composed by 2 sub-workflows:
1. **Physics workflow:** generating events.
2. **ML workflow:** analysing those events.

The combined workflow has this shape:

![image of the workflow](docs/images/workflow-all.png)


## Installation
To install all what is necessary to operate with this project, clone this project as follow:

```shell script
git clone --recurse-submodules https://github.com/scailfin/madminer-workflow
```

For cases where the project has already been cloned:

```shell script
git submodule update --init --recursive
```


## Execution
The full workflow can be launched using [Yadage][yadage-repo]. Yadage is a YAML specification 
language over a set of utilities that are used to coordinate workflows. Please consider that 
it can be hard to define Yadage workflows as the [Yadage documentation][yadage-docs] is incomplete.
For learning about Yadage hidden features contact [Lukas Heinrich][lukas-profile], Yadage creator.

Yadage execution depends on having both Docker environment images (_physics_ and _ML_) already published.
If they are not, please follow the instructions on the [Madminer physics workflow][madminer-workflow-ph]
and [Madminer ML workflow][madminer-workflow-ml] repositories.

Once the Docker images are published:
```shell script
pip3 install yadage
make yadage-run
```


## Execution
This repository is not designed to perform local execution of the combined workflow.
Please, go to the sub-workflow repositories to execute individual steps, 
or [Yadage][yadage-repo] coordinated runs.


## Deployment
To deploy the workflow locally using [REANA][reana-website], install _VirtualBox_ 
as emulator and _Minikube_ as container orchestrator (to simulate a local cluster).
Please refer to the **version 0.7.0** [REANA deployment documentation][reana-deploy-docs]
for details.

To start the workflow:
```shell script
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

It might take some time to finish depending on the job and the cluster.
Once it does, list and download the files:
```shell script
(reana) $ reana-client ls
(reana) $ reana-client download <path/to/file/on/reana/workon>
```


[madminer-workflow-ml]: https://github.com/scailfin/madminer-workflow-ml
[madminer-workflow-ph]: https://github.com/scailfin/madminer-workflow-ph
[reana-website]: http://reanahub.io/
[yadage-repo]: https://github.com/yadage/yadage
[reana-deploy-docs]: http://docs.reana.io/development/deploying-locally/
