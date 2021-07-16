# Ubuntu 18.04 / ROS Melodic Docker Container for Franka Emika Panda
# Copyright 2021 Massachusetts Institute of Technology

# Define the base image
FROM nvidia/cuda:11.1-cudnn8-runtime-ubuntu18.04

# Dependencies before installing ROS and the HSR packages
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    apt-utils \
    build-essential \
    curl \
    locales \
    lsb-release \
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
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt-get update \
 && apt-get install -y --no-install-recommends ros-melodic-desktop-full
RUN apt-get install -y --no-install-recommends python-rosdep
RUN rosdep init \
 && rosdep fix-permissions \
 && rosdep update \
 && echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
 
# Install other packages as needed
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
RUN python3 -m pip install --upgrade pip setuptools
RUN python3 -m pip install wheel
RUN python3 -m pip install \
  catkin_pkg \
  dill \
  empy \
  IPython \
  networkx \
  numba \
  numpy \
  numpy-quaternion \
  psutil \
  pybullet \
  pyquaternion \
  recordclass \
  rospy-message-converter \
  scipy \
  sklearn
RUN python3 -m pip install git+https://github.com/mike-n-7/tsr.git

# Create a Catkin workspace, clone packages, and build using Python 3
RUN source /opt/ros/melodic/setup.bash \
 && mkdir -p /catkin_ws/src \
 && cd /catkin_ws \
 && catkin_init_workspace

# Install Franka packages in the Catkin workspace
RUN apt-get install -y \
    ros-melodic-libfranka \
    ros-melodic-panda-moveit-config
RUN source /opt/ros/melodic/setup.bash \
 && cd /catkin_ws \
 && git clone --recursive https://github.com/frankaemika/franka_ros src/franka_ros \
 && cd src/franka_ros \
 && git checkout tags/0.7.1 -b v0.7.1 \
 && cd ../.. \
 && rosdep install --from-paths src --ignore-src --rosdistro melodic -y

# Install TF and PyKDL packages to work with Python3
RUN source /opt/ros/melodic/setup.bash \
 && cd /catkin_ws/src \
 && git clone -b melodic-devel https://github.com/ros/geometry.git \
 && git clone -b melodic-devel https://github.com/ros/geometry2.git \
 && git clone -b master --recurse-submodules https://github.com/orocos/orocos_kinematics_dynamics.git

# Set up PDDLStream
RUN mkdir -p /external
RUN cd /external \
 && git clone --recurse-submodules https://github.com/caelan/pddlstream.git \
 && cd pddlstream \
 && ./downward/build.py

# Set up stacking repo
RUN git clone https://github.com/Learning-and-Intelligent-Systems/stacking.git \
 && cd stacking \
 && python3 -m pip install --ignore-installed -r requirements.txt 

# Set up pb_robot
# NOTE: You may need to rebuild from this step if there are updates to the repo
RUN source /opt/ros/melodic/setup.bash \
 && cd /external \
 && git clone https://github.com/mike-n-7/pb_robot.git \
 && cd pb_robot/src/pb_robot/ikfast/franka_panda \
 && python3 setup.py build \
 && cd /external/pb_robot/src/pb_robot/models \
 && touch CATKIN_IGNORE

# Add the franka_ros_interface and panda_vision packages
RUN source /opt/ros/melodic/setup.bash \
 && cd /catkin_ws/src \
 && git clone -b melodic https://github.com/rachelholladay/franka_ros_interface \
 && git clone https://github.com/carismoses/panda_vision.git

# Install final ROS dependencies
RUN source /opt/ros/melodic/setup.bash \
 && cd /catkin_ws \
 && rosdep install --from-paths src --ignore-src --rosdistro melodic -y
 
# Add Catkin workspace build and additional paths to the ~/.bashrc
RUN printf "\
if [ ! -f /catkin_ws/devel/setup.bash ]; then \n\
  pushd /catkin_ws \n\
  catkin clean -y \n\
  catkin build -DPYTHON_EXECUTABLE=$(which python3) \n\
  popd \n\
fi \n\
source /catkin_ws/devel/setup.bash \n\
ln -s /external/pb_robot/src/pb_robot .\n\
ln -s /external/pddlstream/pddlstream .\n\
export PYTHONPATH=\$PYTHONPATH:/catkin_ws/src/stacking" >> ~/.bashrc
