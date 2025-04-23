
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install python3-pip -y
RUN pip3 install --upgrade pip
RUN apt-get install tmux -y
RUN apt-get install vim -y
RUN apt install libglfw3-dev -y
RUN apt install libglfw3 -y
RUN apt-get update && apt-get install -y git
RUN pip3 install --upgrade pip setuptools wheel

COPY requirements.txt /tmp/requirements.txt
# Need to use specific cuda versions for jax
ARG USE_CUDA=true
RUN if [ "$USE_CUDA" = true ] ; \
    then pip install "jax[cuda12]>=0.4.25, <0.6.0" -f "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html" ; \
    fi
RUN pip3 install -r /tmp/requirements.txt

ARG WANDB_API
ARG USERNAME
ARG USER_UID
ARG USER_GID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID --create-home $USERNAME \
    && chown -R $USER_UID:$USER_GID /home/$USERNAME

USER $USERNAME

ENV WANDB_API_KEY=$WANDB_API

WORKDIR rl_optimizer/

CMD ["/bin/bash"]
