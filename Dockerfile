# Ubuntu 18.04 / ROS Melodic Docker Container for Franka Emika Panda
# Copyright 2021 Massachusetts Institute of Technology
# Define the base image
FROM nvidia/cuda:11.1-cudnn8-runtime-ubuntu18.04
# Dependencies before installing ROS and the HSR packages
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    apt-utils \
    locales \
    lsb-release \
    build-essential \
 && apt-get clean
# Set up locale and UTF-8 encoding, mostly so setup runs without errors
RUN locale-gen en_US.UTF-8
ENV PYTHONIOENCODING UTF-8
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
RUN dpkg-reconfigure locales
# Install ROS Melodic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt-get update \
 && apt-get install -y --no-install-recommends ros-melodic-desktop-full
RUN apt-get install -y --no-install-recommends python-rosdep
RUN rosdep init \
 && rosdep fix-permissions \
 && rosdep update \
 && echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc

# Install other packages as needed
# RUN sudo apt-get remove -y python3.6
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    cmake \
    ca-certificates \
    g++ \
    g++-multilib \
    git \
    libspatialindex-c4v5 \
    make \
    psmisc \
    python-catkin-pkg \
    python-catkin-pkg-modules \
    python-catkin-tools \
    python-wstool \
    python3.7 \
    python3.7-dev \
    python3-catkin-pkg-modules \
    python3-rospkg-modules \
    python3-dev \
    python3-pip \
    python3-tk \
    python3-venv
# Change the shell to bash so the ROS environment is set up for further instructions
SHELL ["/bin/bash", "-c"]
# Install Python3 packages 
ENV ROS_PYTHON_VERSION 3
RUN python3.7 -m pip install --upgrade pip setuptools
RUN python3.7 -m pip install wheel
RUN python3.7 -m pip install \
  catkin_pkg \
  empy \
  IPython \
  networkx \
  numba \
  numpy \
  numpy-quaternion \
  pybullet \
  recordclass \
  rospy-message-converter \
  scipy
RUN python3.7 -m pip install git+https://github.com/mike-n-7/tsr.git
# Create a Catkin workspace, clone packages, and build using Python 3
RUN source /opt/ros/melodic/setup.bash \
 && mkdir -p /catkin_ws/src \
 && cd /catkin_ws \
 && catkin_init_workspace

# Install franka packages and TF in the Catkin workspace
RUN apt-get install ros-melodic-libfranka
RUN source /opt/ros/melodic/setup.bash \
 && cd /catkin_ws \
 && git clone -b melodic-devel https://github.com/ros/geometry.git src/geometry \
 && git clone -b melodic-devel https://github.com/ros/geometry2.git src/geometry2 \
 && git clone --recursive https://github.com/frankaemika/franka_ros src/franka_ros \
 && cd src/franka_ros \
 && git checkout tags/0.7.1 -b v0.7.1 \
 && cd ../.. \
 && source /opt/ros/melodic/setup.bash \
 && rosdep install --from-paths src --ignore-src --rosdistro melodic -y \
 && catkin build -j1 -DPYTHON_EXECUTABLE=/usr/bin/python3.7
# Set up pb_robot
RUN mkdir -p /external
RUN source /opt/ros/melodic/setup.bash \
 && cd /external \
 && git clone https://github.com/mike-n-7/pb_robot.git \
 && cd pb_robot/src/pb_robot/ikfast/franka_panda \
 && python3.7 setup.py build
# Set up stacking
RUN git clone https://github.com/Learning-and-Intelligent-Systems/stacking.git \
 && cd stacking \
 && python3.7 -m pip install --default-timeout=1000 --ignore-installed -r requirements.txt

# Set up PDDLStream
RUN cd /external \
 && git clone https://github.com/caelan/pddlstream.git \
 && cd pddlstream \
 && git submodule update --init --recursive \
 && ./FastDownward/build.py
# Symlinks to run things from the stacking repo
RUN cd stacking \
 && ln -s /external/pb_robot/src/pb_robot . \
 && ln -s /external/pddlstream/pddlstream .
# Finally, add the franka_ros_interface package
RUN source /opt/ros/melodic/setup.bash \
 && cd /catkin_ws \
 && git clone -b melodic https://github.com/rachelholladay/franka_ros_interface src/franka_ros_interface
# Rebuild the Catkin workspace
RUN source /opt/ros/melodic/setup.bash \
 && cd /catkin_ws \
 && rosdep install --from-paths src --ignore-src --rosdistro melodic -y \
 && catkin build -DPYTHON_EXECUTABLE=/usr/bin/python3.7

 RUN echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc
