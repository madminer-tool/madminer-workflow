YADAGE_TEST_FOLDER="$(PWD)/.yadage"
INPUT_LOCAL_FOLDER="$(PWD)/modules"
WORKFLOW_LOCAL_FOLDER="$(PWD)/reana"


all: yadage-clean yadage-run


.PHONY: yadage-clean
yadage-clean:
	@echo "Cleaning previous run..."
	@rm -rf $(YADAGE_TEST_FOLDER)


.PHONY: yadage-run
yadage-run: yadage-clean
	@echo "Launching Yadage..."
	@yadage-run $(YADAGE_TEST_FOLDER) "workflow.yml" \
		-p input_file_ph="$(WORKFLOW_LOCAL_FOLDER)/ph/input.yml" \
		-p input_file_ml="$(WORKFLOW_LOCAL_FOLDER)/ml/input.yml" \
		-p num_jobs="6" \
		-p train_samples="1"  \
		-d initdir=$(INPUT_LOCAL_FOLDER) \
		--toplevel $(WORKFLOW_LOCAL_FOLDER)
