FROM irinahub/docker-madminer-madgraph:latest

USER root 

RUN apt-get update && apt-get install -y python3-pip python python-pip python3-tk python-tk

RUN	pip3 install six      && \
	pip3 install torch    && \
    pip3 install 'numpy>=1.13.1'  	 && \
    pip install 'numpy>=1.13.1'  	 && \
    pip3 install 'pandas<0.21'  	 && \
    pip install 'pandas<0.21'  	 && \
    pip3 install matplotlib 



RUN pip install --upgrade pip
RUN pip install madminer --upgrade 	&& pip install PyYAML 


WORKDIR /home/
#COPY /madminer ./madminer
COPY /code ./code

RUN chmod 755 -R ./code
#RUN chmod 755 -R ./madminer