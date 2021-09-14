YADAGE_WORKDIR = "$(PWD)/.yadage"

MLFLOW_USERNAME ?= $(shell whoami)
MLFLOW_TRACKING_URI ?= "file:///_mlflow"

WORKFLOW_DIR = "$(PWD)/reana"
WORKFLOW_FILE = "workflow.yml"
WORKFLOW_NAME = "madminer-workflow"

PH_REPOSITORY = "madminer-workflow-ph"
ML_REPOSITORY = "madminer-workflow-ml"


.PHONY: copy
copy:
	@echo "Copying sub-workflows..."
	@cp -r "modules/$(PH_REPOSITORY)/workflow/." "$(WORKFLOW_DIR)/ph"
	@cp -r "modules/$(ML_REPOSITORY)/workflow/." "$(WORKFLOW_DIR)/ml"


.PHONY: reana-check
reana-check:
	@echo "Checking REANA spec..."
	@cd $(WORKFLOW_DIR) && reana-client validate --environments


.PHONY: reana-run
reana-run: copy
	@echo "Deploying on REANA..."
	@cd $(WORKFLOW_DIR) && \
		reana-client create -n $(WORKFLOW_NAME) && \
		reana-client upload -w $(WORKFLOW_NAME) ph && \
		reana-client upload -w $(WORKFLOW_NAME) ml && \
		reana-client upload -w $(WORKFLOW_NAME) $(WORKFLOW_FILE) && \
		reana-client start -w $(WORKFLOW_NAME) \
			-p mlflow_server=$(MLFLOW_TRACKING_URI) \
			-p mlflow_username=$(MLFLOW_USERNAME)


.PHONY: yadage-clean
yadage-clean: copy
	@echo "Cleaning previous run..."
	@rm -rf $(YADAGE_WORKDIR)


.PHONY: yadage-run
yadage-run: yadage-clean
	@echo "Launching Yadage..."
	@yadage-run $(YADAGE_WORKDIR) $(WORKFLOW_FILE) \
		-p input_file_ph="ph/input.yml" \
		-p input_file_ml="ml/input.yml" \
		-p num_procs_per_job="1" \
		-p mlflow_args_s="\"''\"" \
		-p mlflow_args_t="\"''\"" \
		-p mlflow_args_e="\"''\"" \
		-p mlflow_server=$(MLFLOW_TRACKING_URI) \
		-p mlflow_username=$(MLFLOW_USERNAME) \
		-d initdir=$(WORKFLOW_DIR) \
		--toplevel $(WORKFLOW_DIR)
