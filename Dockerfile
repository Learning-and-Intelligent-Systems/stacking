# Use an official Python runtime as a parent image
FROM nvidia/cuda:9.0-runtime-ubuntu16.04

# Install python 3.7
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.7
RUN apt-get install -y python3.7-dev
RUN apt-get install -y python3-pip
RUN apt-get install -y git

# make python 3.7 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.5 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 2
RUN update-alternatives  --set python /usr/bin/python3.7

# Install pb_robot
RUN python3.7 -m pip install numpy pybullet recordclass catkin_pkg IPython networkx scipy
RUN python3.7 -m pip install git+https://github.com/mike-n-7/tsr.git
RUN git clone https://github.com/mike-n-7/pb_robot.git
WORKDIR /pb_robot/src/pb_robot/ikfast/franka_panda
RUN python3.7 setup.py build

# Install stacking dependencies
COPY requirements.txt .
RUN xargs -n1 python3.7 -m pip install --trusted-host pypi.python.org < requirements.txt

# Install pddlstream
WORKDIR /
RUN git clone https://github.com/caelan/pddlstream.git
WORKDIR /pddlstream
RUN git submodule update --init --recursive
RUN apt-get install -y cmake g++ g++-multilib make python
RUN ./FastDownward/build.py

# install minio client
WORKDIR /
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget https://dl.min.io/client/mc/release/linux-amd64/mc && chmod +x mc

# install nano 
RUN apt-get update
RUN apt-get install -y nano

# use this to avoid too much CPU usage
RUN export OMP_NUM_THREADS=1

# add large datasets to docker image
#COPY learning/data/random_blocks_\(x10000\)_2to5blocks_uniform_density.pkl .
#COPY learning/data/random_blocks_\(x2000\)_2to5blocks_uniform_density.pkl .

# run training
COPY train.sh .
RUN chmod +x train.sh
CMD ./train.sh
