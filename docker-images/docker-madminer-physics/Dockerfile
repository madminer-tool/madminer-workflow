#### Base image
#### Reference: https://github.com/diana-hep/madminer/blob/master/Dockerfile
FROM madminertool/docker-madminer:latest


#### Install binary dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    rsync \
    ca-certificates \
    gfortran \
    build-essential \
    ghostscript \
    libboost-all-dev


#### Define working folders
ENV PROJECT_FOLDER "/madminer"
ENV SOFTWARE_FOLDER "/madminer/software"


#### Install MadGraph 5
WORKDIR ${SOFTWARE_FOLDER}

ENV MG_VERSION "MG5_aMC_v2.6.7"
ENV MG_FOLDER "MG5_aMC_v2_6_7"
ENV MG_BINARY "MG5_aMC_v2_6_7/bin/mg5_aMC"
RUN curl -sSL "https://launchpad.net/mg5amcnlo/2.0/2.6.x/+download/${MG_VERSION}.tar.gz" | tar -xzv

# ROOT environment variables
ENV ROOTSYS /usr/local
ENV PATH $PATH:$ROOTSYS/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$ROOTSYS/lib
ENV DYLD_LIBRARY_PATH $DYLD_LIBRARY_PATH:$ROOTSYS/lib

RUN echo "install lhapdf6" | python2 ${SOFTWARE_FOLDER}/${MG_BINARY}
RUN echo "install pythia8" | python2 ${SOFTWARE_FOLDER}/${MG_BINARY}
RUN echo "install Delphes" | python2 ${SOFTWARE_FOLDER}/${MG_BINARY}

# Delphes environment variables
ENV ROOT_INCLUDE_PATH $ROOT_INCLUDE_PATH:${SOFTWARE_FOLDER}/${MG_FOLDER}/Delphes/external


#### Set working directory
WORKDIR ${PROJECT_FOLDER}

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
