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
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --requirement requirements.txt

# Install Python3 dependencies
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir --requirement requirements.txt
