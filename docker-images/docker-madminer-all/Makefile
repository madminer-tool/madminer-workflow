NAME=docker-madminer-all
ML_FOLDER=docker-madminer-ml
PH_FOLDER=docker-madminer-physics
REGISTRY=madminertool
VERSION=$(shell cat ./VERSION)


all: build copy push


.PHONY: copy
copy:
	@cp ../$(ML_FOLDER)/requirements.txt .
	@cp -R ../$(ML_FOLDER)/code/ code/
	@cp -R ../$(PH_FOLDER)/code/ code/


.PHONY: build
build: copy
	@docker build . --tag $(REGISTRY)/$(NAME):$(VERSION) --tag $(REGISTRY)/$(NAME):latest


.PHONY: push
push: build
	@docker login --username "${DOCKERUSER}" --password "${DOCKERPASS}"
	@docker push $(REGISTRY)/$(NAME):$(VERSION)
	@docker push $(REGISTRY)/$(NAME):latest
