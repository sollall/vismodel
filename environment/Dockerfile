FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get -y install curl git htop vim wget zip nodejs zip build-essential python3 python3.10 python3-pip  

# add conda
RUN pip install --upgrade pip && \
    pip install --upgrade setuptools && \
    pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121

CMD ["/bin/bash"]
