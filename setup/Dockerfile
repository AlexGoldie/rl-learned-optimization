
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install python3-pip -y
RUN pip3 install --upgrade pip
RUN apt-get install tmux -y
RUN apt-get install vim -y
RUN apt install libglfw3-dev -y
RUN apt install libglfw3 -y

# install dependencies via pip
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

ENV WANDB_API_KEY=<FILL THIS IN>

WORKDIR rl_optimizer/

CMD ["/bin/bash"]
