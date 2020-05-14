NAME=docker-madminer-ml
REGISTRY=madminertool
VERSION=$(shell cat ./VERSION)


all: clean build push


.PHONY: clean
clean:
	@rm -rf data
	@rm -rf models
	@rm -rf plots
	@rm -rf rates
	@rm -rf results
	@rm -rf test
	@find . -name "*.tar.gz" -delete


.PHONY: build
build:
	@docker build . --tag $(REGISTRY)/$(NAME):$(VERSION) --tag $(REGISTRY)/$(NAME):latest


.PHONY: push
push: build
	@docker login --username "${DOCKERUSER}" --password "${DOCKERPASS}"
	@docker push $(REGISTRY)/$(NAME):$(VERSION)
	@docker push $(REGISTRY)/$(NAME):latest
