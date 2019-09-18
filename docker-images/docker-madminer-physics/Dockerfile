###############################
#
# Dockerfile for MG5 Pythia Delphes and patches for Madminer deployment
#

FROM madminertool/docker-madminer:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates python python3 gfortran build-essential ghostscript vim libboost-all-dev


#
# MadGraph + Pythia + Delphes
#



WORKDIR /madminer/software

ENV MG_VERSION="MG5_aMC_v2_6_2" 

COPY ${MG_VERSION}/ ./${MG_VERSION}
RUN ./${MG_VERSION}/bin/mg5_aMC

#config path
WORKDIR /madminer/software/${MG_VERSION}
ENV ROOTSYS /usr/local 
ENV PATH $PATH:$ROOTSYS/bin 
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$ROOTSYS/lib

RUN echo "install lhapdf6" | /madminer/software/${MG_VERSION}/bin/mg5_aMC
RUN echo "install pythia8" | /madminer/software/${MG_VERSION}/bin/mg5_aMC
RUN echo "install pythia-pgs" | /madminer/software/${MG_VERSION}/bin/mg5_aMC
RUN echo "install Delphes" | /madminer/software/${MG_VERSION}/bin/mg5_aMC



#
# patches for Madminer
#

WORKDIR /madminer/software

#add patches folder
COPY patch/ ./patch

#running patch.py requires being in the directory
WORKDIR /madminer/software/patch

ENV MG_PATH /madminer/software/${MG_VERSION}

#run patches with MG directory
RUN python patch.py ${MG_PATH}


#extra python library
RUN pip install PyYAML 

#
#CODE FOR THE PHYSICS WORKFLOW
#
WORKDIR /madminer
COPY /code ./code

RUN chmod 755 -R ./code

WORKDIR /madminer