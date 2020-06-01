#### Base image
#### Reference: https://github.com/diana-hep/madminer/blob/master/Dockerfile
FROM madminertool/docker-madminer:latest


#### Set working directory
WORKDIR /madminer

#### Copy files
COPY code ./code
COPY scripts ./scripts
COPY requirements.txt .


# Install Python2 dependencies
RUN python2 -m pip install --upgrade pip && \
    python2 -m pip install --no-cache-dir --requirement requirements.txt

# Install Python3 dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir --requirement requirements.txt
