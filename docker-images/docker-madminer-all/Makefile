NAME=docker-madminer-all
ML_FOLDER=docker-madminer-ml
PH_FOLDER=docker-madminer-physics
REGISTRY=madminertool
VERSION=$(shell cat ./VERSION)


all: build clean copy push


.PHONY: clean
clean:
	@rm -rf code
	@find . -name "requirements-*.txt" -delete


.PHONY: copy
copy:
	@cp -R ../$(ML_FOLDER)/code/ code/
	@cp -R ../$(PH_FOLDER)/code/ code/
	@cp ../$(ML_FOLDER)/requirements.txt requirements-ml.txt
	@cp ../$(PH_FOLDER)/requirements.txt requirements-ph.txt


.PHONY: build
build: copy
	@docker build . --tag $(REGISTRY)/$(NAME):$(VERSION) --tag $(REGISTRY)/$(NAME):latest


.PHONY: push
push: build
	@docker login --username "${DOCKERUSER}" --password "${DOCKERPASS}"
	@docker push $(REGISTRY)/$(NAME):$(VERSION)
	@docker push $(REGISTRY)/$(NAME):latest
