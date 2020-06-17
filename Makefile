YADAGE_TEST_FOLDER="$(PWD)/.yadage"
INPUT_LOCAL_FOLDER="$(PWD)/reana/inputs"
WORKFLOW_LOCAL_FOLDER="$(PWD)/reana/workflows/yadage"


all: clean run


.PHONY: clean
clean:
	@echo "Cleaning previous execution..."
	@rm -rf $(YADAGE_TEST_FOLDER)


.PHONY: run
run: clean
	@echo "Launching Yadage..."
	@yadage-run $(YADAGE_TEST_FOLDER) "workflow.yml" \
		-p inputfile="input.yml" \
		-p njobs="6" \
		-p ntrainsamples="1"  \
		-d initdir=$(INPUT_LOCAL_FOLDER) \
		--toplevel $(WORKFLOW_LOCAL_FOLDER) \
		--visualize
