FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get -y install curl git htop vim wget zip nodejs zip build-essential

# add conda
WORKDIR /opt
ENV PATH /opt/miniconda3/bin:$PATH
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    sh Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 && \
    rm -r Miniconda3-latest-Linux-x86_64.sh && \
    conda init && conda create -n vismodel python=3.10

WORKDIR /workspace

CMD ["/bin/bash"]
