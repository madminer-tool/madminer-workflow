YADAGE_WORKDIR="$(PWD)/.yadage"

MLFLOW_TRACKING_URI ?= "/tmp/mlflow"

WORKFLOW_FOLDER="$(PWD)/reana"
WORKFLOW_NAME="madminer-workflow"

PH_REPOSITORY="madminer-workflow-ph"
ML_REPOSITORY="madminer-workflow-ml"


all: copy reana-deploy yadage-clean yadage-run


.PHONY: copy
copy:
	@echo "Copying sub-workflows..."
	@cp -r "modules/$(PH_REPOSITORY)/workflow/." "$(WORKFLOW_FOLDER)/ph"
	@cp -r "modules/$(ML_REPOSITORY)/workflow/." "$(WORKFLOW_FOLDER)/ml"


.PHONY: reana-deploy
reana-deploy: copy
	@echo "Deploying on REANA..."
	@cd $(WORKFLOW_FOLDER) && \
		reana-client create -n $(WORKFLOW_NAME) && \
		reana-client upload -w $(WORKFLOW_NAME) ph && \
		reana-client upload -w $(WORKFLOW_NAME) ml && \
		reana-client upload -w $(WORKFLOW_NAME) workflow.yml && \
		reana-client start -w $(WORKFLOW_NAME)


.PHONY: yadage-clean
yadage-clean: copy
	@echo "Cleaning previous run..."
	@rm -rf $(YADAGE_WORKDIR)


.PHONY: yadage-run
yadage-run: yadage-clean
	@echo "Launching Yadage..."
	@yadage-run $(YADAGE_WORKDIR) "workflow.yml" \
		-p input_file_ph="ph/input.yml" \
		-p input_file_ml="ml/input.yml" \
		-p num_jobs="6" \
		-p mlflow_server=$(MLFLOW_TRACKING_URI) \
		-d initdir=$(WORKFLOW_FOLDER) \
		--toplevel $(WORKFLOW_FOLDER)
