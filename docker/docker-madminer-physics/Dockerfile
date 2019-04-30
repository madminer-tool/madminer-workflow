###############################
#
# Dockerfile for MG5 Pythia Delphes and patches for Madminer deployment
#

FROM madminertool/docker-madminer:latest

USER root 

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates python gfortran build-essential ghostscript vim libboost-all-dev


#
# MadGraph + Pythia + Delphes
#

WORKDIR /home/software

ENV MG_VERSION="MG5_aMC_v2_6_2" 

COPY ${MG_VERSION}/ ./${MG_VERSION}
RUN ./${MG_VERSION}/bin/mg5_aMC

#config path
WORKDIR /home/software/${MG_VERSION}
ENV ROOTSYS /usr/local 
ENV PATH $PATH:$ROOTSYS/bin 
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$ROOTSYS/lib

RUN echo "install lhapdf6" | /home/software/${MG_VERSION}/bin/mg5_aMC
RUN echo "install pythia8" | /home/software/${MG_VERSION}/bin/mg5_aMC
RUN echo "install pythia-pgs" | /home/software/${MG_VERSION}/bin/mg5_aMC
RUN echo "install Delphes" | /home/software/${MG_VERSION}/bin/mg5_aMC



#
# patches for Madminer
#

WORKDIR /home/software

#add patches folder
COPY patch/ ./patch

#running patch.py requires being in the directory
WORKDIR /home/software/patch

ENV MG_PATH /home/software/${MG_VERSION}

#run patches with MG directory
RUN python patch.py ${MG_PATH}


#extra python library
RUN pip install PyYAML 

#
#CODE FOR THE PHYSICS WORKFLOW
#
WORKDIR /home
COPY /code ./code

RUN chmod 755 -R ./code

WORKDIR /home
