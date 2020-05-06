NAME=docker-madminer-physics
REGISTRY=madminertool
VERSION=$(shell cat ./VERSION)


all: clean build push


.PHONY: clean
clean:
	@rm -rf data
	@rm -rf extract
	@rm -rf code/logs
	@rm -rf code/mg_processes
	@find . -name "folder_*.tar.gz" -delete


.PHONY: build
build:
	@docker build . --tag $(REGISTRY)/$(NAME):$(VERSION) --tag $(REGISTRY)/$(NAME):latest


.PHONY: push
push: build
	@docker login --username "${DOCKERUSER}" --password "${DOCKERPASS}"
	@docker push $(REGISTRY)/$(NAME):$(VERSION)
	@docker push $(REGISTRY)/$(NAME):latest
