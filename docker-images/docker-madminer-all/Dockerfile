### NOTE:
### This Docker image takes the "heavy-weight" image
### from the Physics use case as base image
### and complete it with the Docker commands
### from the Machine Learning use case


#### Base image
#### Reference: https://github.com/scailfin/workflow-madminer/blob/master/docker-images/docker-madminer-physics/Dockerfile
FROM madminertool/docker-madminer-physics:latest


#### Set working directory and copy files
WORKDIR /madminer
COPY requirements.txt .
COPY code ./code


# Install Python2 dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --requirement requirements.txt

# Install Python3 dependencies
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir --requirement requirements.txt
