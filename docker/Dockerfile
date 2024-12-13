FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ARG USER_NAME=<username>
ARG UID=<userid>
ARG GROUP_NAME=<groupname>
ARG GID=<groupid>

ENV CONDA_ENV_NAME=devel
ENV PYTHON_VERSION=3.10.13
ENV DEBIAN_FRONTEND noninteractive

# Update the system and install required packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y sudo git apt-utils htop wget screen libgl1 libglib2.0-0

# Create a new group with the same GID as the host
RUN groupadd --gid $GID $GROUP_NAME
# Create a new user with the same UID and GID as the host
RUN useradd --uid $UID --gid $GID --shell /bin/bash --create-home $USER_NAME

# don't require password with sudo, for convenience
# not the safest thing to do, but hopefully okay inside the container
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Add the new user to the sudo group
RUN usermod -aG sudo ${USER_NAME}

USER $USER_NAME

# Install Miniconda and create main env
ADD --chown=${USER_NAME}:${GROUP_NAME} https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh /home/${USER_NAME}/miniconda3.sh 
RUN /bin/bash /home/${USER_NAME}/miniconda3.sh -b -p /home/${USER_NAME}/conda \
    && echo export PATH=/home/${USER_NAME}/conda/bin:$PATH >> /home/${USER_NAME}/bashrc \
    && rm /home/${USER_NAME}/miniconda3.sh

ENV PATH="/home/${USER_NAME}/conda/bin:${PATH}"
RUN conda create -n ${CONDA_ENV_NAME} -y python=${PYTHON_VERSION} 

# Switch to bash shell
SHELL ["/bin/bash", "-c"]

WORKDIR /home/${USER_NAME}

RUN source activate ${CONDA_ENV_NAME} \
    && pip3 install --upgrade pip \
    && pip install ftfy regex future tensorboard tensorboardX wandb torch-tb-profiler chardet \
    && conda install -y pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia \
    && pip3 install -U openmim \
    && mim install mmengine \
    && mim install "mmcv>=2.0.0"

RUN git clone -b main https://github.com/guillem-ms/cloth_seg.git
WORKDIR /home/${USER_NAME}/cloth_seg
RUN source activate ${CONDA_ENV_NAME} \
    && pip install -r requirements.txt \
    && pip install --no-cache-dir -e . \
    && pip install "numpy<2.0"

RUN echo "source activate ${CONDA_ENV_NAME}" >> /home/${USER_NAME}/.bashrc
ENV DEBIAN_FRONTEND teletype