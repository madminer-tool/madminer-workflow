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


## Development
To install all the source code that is necessary to operate with this project:

```shell script
git clone --recurse-submodules https://github.com/madminer-tool/madminer-workflow
```

For cases where the project has already been cloned:

```shell script
git submodule update --init --recursive
```

The repositories defined as sub-modules will follow their own development pace.
For cases where the sub-module repositories has been updated on GitHub, and want
to propagate those changes to your local copy of the repositories:

```shell script
git submodule update --remote
```


## MLFlow ♻️
The [MLFlow][mlflow-website] framework has been integrated with some steps of the workflow
in order to keep track of runs initial set of parameters, set of results, and generated artifacts.

In order to locally deploy your own:
```shell script
# Deploy local tracking server
mlflow server \                                                 
    --host "0.0.0.0" \
    --port 5000 \
    --workers 2 \
    --backend-store-uri "file:///tmp/mlflow/runs/metadata" \
    --default-artifact-root "file:///tmp/mlflow/runs/artifacts"

# Specify server URL to interact with it
export MLFLOW_TRACKING_URI="http://0.0.0.0:5000"

# Create experiments to avoid race conditions on parallelized steps.
mlflow experiments create --experiment-name "madminer-ml-sample"
mlflow experiments create --experiment-name "madminer-ml-train"
mlflow experiments create --experiment-name "madminer-ml-eval"
```


## Execution
The full workflow can be launched using [Yadage][yadage-repo]. Yadage is a YAML specification
language over a set of utilities that are used to coordinate workflows. Please consider that
it can be hard to define Yadage workflows as the [Yadage documentation][yadage-docs] is incomplete.
For learning about Yadage hidden features contact [Lukas Heinrich][lukas-profile], Yadage creator.

Yadage execution depends on having both Docker environment images (_physics_ and _ML_) already pushed.
If they are not, please follow the instructions on the [Madminer physics workflow][madminer-workflow-ph]
and [Madminer ML workflow][madminer-workflow-ml] repositories.

Once the Docker images are available on DockerHub, run locally:
```shell script
export MLFLOW_TRACKING_URI="http://host.docker.internal:5000"
export PACKTIVITY_DOCKER_CMD_MOD="--add-host host.docker.internal:host-gateway"  # Linux only
make yadage-run
```


## Deployment

### Local debugging
To debug the workflow locally using [REANA][reana-website] first install _Docker_
and the `kind` CLI tool (_Kubernetes in Docker_) to deploy a local cluster.
Please follow the [local deployment documentation][reana-deploy-docs] to set up REANA.

To start the workflow:
```shell script
$ source ~/.virtualenvs/reana/bin/activate
(reana) $ eval $(reana-dev client-setup-environment)
(reana) $ export REANA_WORKON=madminer-workflow
(reana) $ export MLFLOW_TRACKING_URI=http://host.docker.internal:5000
(reana) $ make reana-run
```

### Remote deployment
In case you have access to a remote REANA cluster and want to deploy there,
you would need to set up the environment variables yourself:

```shell script
$ source ~/.virtualenvs/reana/bin/activate
(reana) $ export REANA_ACCESS_TOKEN=[..]
(reana) $ export REANA_SERVER_URL=[..]
(reana) $ export REANA_WORKON=madminer-workflow
(reana) $ export MLFLOW_TRACKING_URI=<tracking_server_url>
(reana) $ make reana-run
```

It might take some time to finish depending on the job and the cluster.
Once it does, list and download the files:
```shell script
(reana) $ reana-client ls
(reana) $ reana-client download <path/to/file/on/reana/workon>
```


[lukas-profile]: https://github.com/lukasheinrich
[madminer-workflow-ml]: https://github.com/madminer-tool/madminer-workflow-ml
[madminer-workflow-ph]: https://github.com/madminer-tool/madminer-workflow-ph
[mlflow-website]: https://www.mlflow.org/
[reana-deploy-docs]: http://docs.reana.io/administration/deployment/deploying-locally/
[reana-website]: http://reanahub.io/
[yadage-repo]: https://github.com/yadage/yadage
[yadage-docs]: https://yadage.readthedocs.io/en/latest/
