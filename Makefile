YADAGE_TEST_FOLDER="$(PWD)/.yadage"
WORKFLOW_LOCAL_FOLDER="$(PWD)/reana"

PH_REPOSITORY="madminer-workflow-ph"
ML_REPOSITORY="madminer-workflow-ml"


all: copy yadage-clean yadage-run


.PHONY: copy
copy:
	@echo "Copying sub-workflows..."
	@cp -r "modules/$(PH_REPOSITORY)/workflow/." "$(WORKFLOW_LOCAL_FOLDER)/ph"
	@cp -r "modules/$(ML_REPOSITORY)/workflow/." "$(WORKFLOW_LOCAL_FOLDER)/ml"


.PHONY: yadage-clean
yadage-clean: copy
	@echo "Cleaning previous run..."
	@rm -rf $(YADAGE_TEST_FOLDER)


.PHONY: yadage-run
yadage-run: yadage-clean
	@echo "Launching Yadage..."
	@yadage-run $(YADAGE_TEST_FOLDER) "workflow.yml" \
		-p input_file_ph="ph/input.yml" \
		-p input_file_ml="ml/input.yml" \
		-p num_jobs="6" \
		-p train_samples="1"  \
		-d initdir=$(WORKFLOW_LOCAL_FOLDER) \
		--toplevel $(WORKFLOW_LOCAL_FOLDER)
