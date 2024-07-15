FROM nvidia/cuda:11.7.1-base-ubuntu20.04

# Since wget is missing
RUN apt-get update && apt-get install -y wget

#Install MINICONDA
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda.sh && \
	/bin/bash Miniconda.sh -b -p /opt/conda && \
	rm Miniconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc libsndfile1

# Install gcc as it is missing in our base layer
RUN apt-get update && apt-get -y install gcc

#  Create conda env
RUN conda config --set unsatisfiable_hints false

EXPOSE 5000

# Service specific commands
ENV PYTHONPATH /berte-ai-service:$PYTHONPATH

WORKDIR /berte-ai-service

ADD ./src/main /berte-ai-service/src/main
ADD environment.yaml /berte-ai-service/environment.yaml
ADD pyproject.toml /berte-ai-service/pyproject.toml

# Make required directories
RUN mkdir -p /berte-ai-service/data/logs
RUN mkdir -p /berte-ai-service/data/models
RUN mkdir -p /berte-ai-service/logs

RUN conda env create -f environment.yaml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "berte-ai-service", "/bin/bash", "-c"]

# The code to run when container is started:
ENTRYPOINT ["conda", "run", "-n", "berte-ai-service", "python", "src/main/app/app.py"]